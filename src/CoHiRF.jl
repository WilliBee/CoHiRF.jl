module CoHiRF

using Clustering
using Distances
using OrderedCollections
using Random
using StatsBase
using ParallelKMeans
using Base.Threads

include("kmeans_ninit.jl")

export CoHiRFResult, run_cohirf_iterations, kmeans_ninit

struct CoHiRFResult
    labels::Vector{Int}
    n_clusters::Int
    labels_sequence::Matrix{Int}
    cluster_representatives::Vector{Int}
    n_iterations::Int
end

"""
    align_labels!(labels::AbstractVector{Int})

Remap cluster labels to canonical ordering based on first occurrence.

Assigns new labels sequentially (1, 2, 3,...) in the order they first appear in the
input vector. This ensures deterministic labeling regardless of initial cluster IDs.

# Arguments
- `labels`: Vector of cluster assignments (modified in-place)
"""
function align_labels!(labels::AbstractVector{Int})
    unique_labels = unique(labels)
    label_map = Dict{Int, Int}()

    # First pass: map each unique label to a sequential ID based on first appearance
    for (label_ordered, label_unordered) in enumerate(unique_labels)
        label_map[label_unordered] = label_ordered
    end
    
    # Second pass: apply the mapping to all labels in-place
    for i in eachindex(labels)
        labels[i] = label_map[labels[i]]
    end
end

"""
    build_consensus_matrix!(P_buffer, X;
                            R, q, C, n_init, rng, n_threads) -> P

Build a consensus matrix by running K-Means `R` times on random feature subsets.

Each row of `P` stores cluster assignments from one K-Means run, columns correspond to samples.
Identifies stable cluster structure robust to feature sampling.

# Arguments
- `P_buffer`: Buffer for output consensus matrix (size ≥ R × n_curr)
- `X`: Data matrix (n_features × n_curr)
- `R`: Number of K-Means runs
- `q`: Features per run (if q < p, otherwise uses all)
- `C`: Number of clusters
- `n_init`: Random initializations per run (best result kept)
- `rng`: RNG for reproducibility

# Returns
- `P`: Built consensus matrix
"""
function build_consensus_matrix!(P_buffer::AbstractMatrix{Int},
                                X::AbstractArray;
                                R::Int, 
                                q::Int, 
                                C::Int,
                                n_init::Int=10,
                                rng=Random.GLOBAL_RNG)
    p, n_current = size(X)
    
    # Create view into output buffer
    P = @view P_buffer[:, 1:n_current]
    
    # Run R independent K-Means with random feature subsets
    for r in 1:R
        # Sample random features for this run
        features = sample(rng, 1:p, min(q, p); replace=false)

        result = kmeans_ninit(
            X[features, :], 
            min(C, n_current), 
            n_init=n_init, 
            rng=rng
            )
        align_labels!(result)
        P[r, :] .= result
    end
    
    return P
end


"""
    find_consensus_groups!(cluster_membership::OrderedDict, P::AbstractMatrix{Int}) -> cluster_membership

Group samples with identical clustering histories.
Identifies samples that were assigned to the same clusters across all `R` runs.
These "consensus groups" represent perfectly stable clustering patterns.

# Arguments
- `cluster_membership`: Mapping between consensus codes and member indices. The order 
  of insertion matches the order codes are first encountered.
- `P`: Consensus matrix of shape `(n_runs, n_samples)`. Each column encodes a sample's 
  cluster assignments across runs.

# Returns
- `cluster_membership`: Maps `Tuple(code)` → `[sample_indices]`
"""
function find_consensus_groups!(cluster_membership::OrderedDict, P::AbstractMatrix{Int})
    empty!(cluster_membership)

    for i in axes(P, 2)
        code = Tuple(@view P[:, i])
        # Get or create the set for this code, then add the current sample index
        push!(
            get!(cluster_membership, code, Int[]),
            i
        )
    end
end

"""
    propagate_labels!(label_sequence_e, cluster_membership, children_of_X_j) -> label_seq

Propagate cluster assignments from compressed to original sample space.

Each compressed sample (in X_j) represents multiple original samples (in X). This 
function assigns the cluster ID of each compressed sample to all original samples 
it represents.

# Arguments
- `label_sequence_e`: Vector of length n containing cluster assignment of each original sample in X at current iteration e
- `cluster_membership`: OrderedDict(key = code, value = array of X_j indices) represents the groupings of rows of X_j into clusters
- `children_of_X_j`: Mapping `compressed_sample_index => [original_sample_indices]`

# Returns (in-place)
- `label_sequence_e`: The modified labels vector
"""
function propagate_labels!(
    label_sequence_e, 
    cluster_membership, 
    children_of_X_j
    )
    for (cluster_idx, (_, consensus_group_members)) in enumerate(cluster_membership)
        for m in consensus_group_members
            for i in children_of_X_j[m]
                label_sequence_e[i] = cluster_idx
            end
        end
    end
end

"""
    compute_medoid(X::AbstractMatrix, metric) -> medoid_idx

Find the medoid (sample that minimizes total pairwise distance) of the columns of `X`.

# Arguments
- `X`: Data matrix where each column is a sample
- `metric`: Distance function that accepts two column views (e.g., `Euclidean()`)

# Returns
- `medoid_idx`: Index of the column that is the medoid
"""
function compute_medoid(X::AbstractArray, distance_func)
    local_cluster_distances = pairwise(distance_func, X, dims=2)
    local_cluster_distances_sum = sum(local_cluster_distances, dims=2)
    argmin(local_cluster_distances_sum)[1]
end


"""
    assign_medoids!(buff_X_j, buff_cluster_reps, children, buff_dist, X_j, 
                    cluster_membership, label_seq, cluster_reps) -> new_X_j, new_cluster_reps, new_children

Replace each cluster with its medoid and update hierarchical mapping.

For each consensus group, finds the sample (medoid) that minimizes total distance
to group members. Rebuilds the compressed representation using medoids only.


# Arguments
- `children_of_X_j`: Dict of Int => Vectors{Int}; Vector at key i stores original indices of samples represented by medoid i
- `X_j`: Current data matrix (features × samples)
- label_sequence_e : Cluuster assignment of each sample in original X at current iteration e
- `cluster_reps`: Global indices of samples in X_j (maps X_j columns back to original X)
- `cluster_membership`: Dict mapping consensus pattern → local member indices
- `metric`: Distance function that accepts two column views (e.g., `Euclidean()`)

# Returns
- `new_X_j`: Matrix where each column is a cluster medoid
- `new_cluster_reps`: Global indices of selected medoids
"""
function assign_medoids!(
    children_of_X_j::AbstractDict, 
    X_j::AbstractMatrix, 
    n_curr::Int, 
    label_sequence_e::AbstractVector{Int}, 
    cluster_reps::AbstractVector{Int}, 
    cluster_membership::OrderedDict,
    metric=CosineDist()
    )
    
    m = size(X_j, 1)
    n_curr = length(cluster_membership)

    # Clear and reinitialize children mapping for this iteration
    empty!(children_of_X_j)

    # Pre-allocate outputs
    new_X_j = Matrix{eltype(X_j)}(undef, m, n_curr)
    new_cluster_reps = Vector{Int}(undef, n_curr)
    
    # Process each consensus group
    for (cluster_idx, (_, in_xj_members_indices)) in enumerate(cluster_membership)
        # 1. Find medoid within group's local indices
        in_cluster_medoid_index = compute_medoid(X_j[:, in_xj_members_indices], metric)
 
        # 2. Convert to global index in original data X
        global_medoid_index = cluster_reps[in_xj_members_indices[in_cluster_medoid_index]]
        
        # 3. Store medoid features and global index
        new_X_j[:, cluster_idx] .= @view X_j[:, in_xj_members_indices[in_cluster_medoid_index]]
        new_cluster_reps[cluster_idx] = global_medoid_index

        # 4. Update children mapping: medoid → all members in original space
        global_members_indices = findall(==(cluster_idx), label_sequence_e)
        children_of_X_j[cluster_idx] = global_members_indices
    end

    return new_X_j, new_cluster_reps
end


"""
    run_cohirf_iterations(X; q, R, C, max_iter, output_label_seq) -> CoHiRFResult

Run the Co-Hierarchical Random Forest (CoHiRF) clustering algorithm.

Performs iterative consensus clustering using random feature subsets. At each iteration:
1. Build consensus matrix from multiple K-Means runs on random features
2. Identify stable consensus groups (samples with identical clustering histories)
3. Replace groups with their medoids to compress the data
4. Repeat until convergence (number of groups stabilizes)

**Parameters**:
- `X`: Data matrix (n_features × n_samples)
- `q`: Number of features to sample per K-Means run
- `R`: Number of K-Means runs per iteration
- `C`: Number of clusters for K-Means
- `max_iter`: Maximum iterations before early stopping
- `n_init`: Random initializations per run (best result kept)
- `metric`: Distance function that accepts two column views (e.g., `Euclidean()`)
- `output_label_seq`: If true, store full label history for analysis

**Returns**:
- `CoHiRFResult` struct containing:
  - `labels`: Final cluster assignments
  - `n_clusters`: Number of clusters found
  - `labels_sequence`: History of labels per iteration (if output_label_seq=true)
  - `cluster_representatives`: Global indices of final medoids
  - `n_iterations`: Total iterations executed

**Notes**:
- Buffers are pre-allocated for performance and reused across iterations
- Convergence detected when consensus group count stops changing
- Hierarchical structure tracked via `children_of_X_j` mapping
"""
function run_cohirf_iterations(
    X::AbstractArray{T};
    q::Int, 
    R::Int, 
    C::Int,
    max_iter::Int=300,
    n_init::Int=10,
    metric=Euclidean(),
    output_label_seq::Bool=false
    ) where T
    
    n = size(X, 2)

    # Track label evolution of each sample in X across iterations (optional)
    labels_sequence = Matrix{Int}(undef, n, 0)

    # Cluster label of samples in X at current iteration e
    label_sequence_e = collect(1:n)

    # Hierarchical mapping: compressed index -> original sample indices
    children_of_X_j = Dict{Int, Vector{Int}}(i => [i] for i in 1:n)

    # Global indices of current medoids (maps X_j columns back to original X)
    cluster_representatives = collect(1:n)

    # Current working data (starts as original, gets compressed each iteration)
    X_j = X
    n_prev = 0
    n_curr = n
    iteration = 1

    # Pre-allocated buffers
    P_buffer = zeros(Int, R, n)
    cluster_membership = OrderedDict{Tuple, Vector{Int}}()

    while iteration <= max_iter
        # Step 1: Build consensus matrix from R K-Means runs on feature subsets
        # P[r, i] = cluster assignment of sample i in run r
        P = build_consensus_matrix!(P_buffer, X_j, R=R, C=C, q=q, n_init=n_init)
        
        # Step 2: Find consensus groups = samples with identical rows in P
        # Each group represents a stable cluster across all feature subsamples
        find_consensus_groups!(cluster_membership, P)

        # Track convergence: compare number of groups to previous iteration
        n_prev = n_curr
        n_curr = length(cluster_membership)
        
        if n_curr == n_prev
            break
        end

        # Step 3: Propagate labels from consensus groups to original samples
        # Assign each sample the cluster ID of its medoid
        propagate_labels!(label_sequence_e, cluster_membership, children_of_X_j)

        # Store full history for analysis (if requested)
        if output_label_seq
            labels_sequence = hcat(labels_sequence, label_sequence_e)
        end

        # Step 4: Compress data by replacing groups with their medoids
        # X_j becomes (features × n_curr), shrinking each iteration
        X_j, cluster_representatives = assign_medoids!(
            children_of_X_j, X_j, n_curr, label_sequence_e, 
            cluster_representatives, cluster_membership, metric
        )
        
        iteration += 1
    end

    return CoHiRFResult(
        label_sequence_e,                   # Final cluster assignments
        length(unique(label_sequence_e)),   # Number of clusters
        labels_sequence,                    # Full iteration history
        cluster_representatives,            # Indices of final medoids
        iteration - 1                       # Actual iterations completed
    )
end


end # module CoHiRF
