"""
    kmeans_ninit(X, k; n_init, max_iters, tol, k_init, rng, n_threads, alg) -> best_assignments

Run K-Means multiple times and return the best clustering result.

Performs `n_init` independent K-Means runs with different random initializations
and returns the cluster assignments with the lowest inertia (cost). 

**Parameters**:
- `X`: Data matrix (n_features × n_samples)
- `k`: Number of clusters
- `n_init`: Number of independent runs
- `max_iters`: Max iterations per run
- `tol`: Convergence tolerance for cost improvement
- `k_init`: Initialization strategy ("k-means++" or "random")
- `rng`: Random number generator for reproducibility
- `n_threads`: Number of threads for ParallelKMeans.jl
- `alg`: K-Means algorithm backend (Yinyang is fast for large k)

**Returns**:
- `best_assignments::Vector{Int}`: Cluster assignments for best run (length = n_samples)
"""

function kmeans_ninit(
    X::AbstractMatrix{<:Real},
    k::Integer;
    n_init::Integer = 10,
    max_iters::Integer = 300,
    tol = 1e-6,
    k_init = "k-means++",
    rng::Random.AbstractRNG = Random.GLOBAL_RNG,
    n_threads::Int = 1,
    alg = Hamerly()
)
    best_assignments = nothing
    best_inertia = Inf

    # Pre-allocate thread-safe K-Means containers (reused across runs)
    containers = ParallelKMeans.create_containers(
        alg, X, k, size(X, 1), size(X, 2), 1
    )

    for _ in 1:n_init
        result = ParallelKMeans.kmeans!(
            alg, containers, X, k, nothing,
            n_threads = n_threads,
            k_init = k_init, 
            max_iters = max_iters,
            tol = tol,
            init = nothing,
            rng=rng)
        inertia =  result.totalcost
        
        if inertia < best_inertia
            best_inertia = inertia
            best_assignments = result.assignments
        end
    end

    return best_assignments
end