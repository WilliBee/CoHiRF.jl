using BenchmarkTools
using Random
using Clustering
using CoHiRF
import CoHiRF: kmeans_ninit

# =========== Scalability Tests ===============

# PARAMETERS
q = 10
R = 10
C = 3

grid = [100, 347, 1202, 4163, 14427, 50000]
n_clusters = 5

# DATA GENERATION
function generate_data(n_features, n_samples, n_clusters, Δ=100.0)
    centres = [rand(0:1, n_features) .* Δ for _ in 1:n_clusters]
    y = rand(1:n_clusters, n_samples)
    X = Matrix{Float64}(undef, n_features, n_samples)
    for (i, c) in enumerate(y)
        X[:, i] .= randn(n_features) .+ centres[c]
    end
    return X, y
end

sizes_to_test = ((grid[5], el) for el in grid)
sizes_to_test = vcat(sizes_to_test..., ((b,a) for (a,b) in sizes_to_test)...)

# EXECUTION
for (n_samples, n_features) in sizes_to_test
    X, y = generate_data(n_features, n_samples, n_clusters)

    cohirf_results = run_cohirf_iterations(X, R=R, C=C, q=q, max_iter=300)
    labels = cohirf_results.labels
    
    @info "Generated" n_samples n_features
    @show cohirf_ari = randindex(y, labels)[1]

    # BENCHMARKING
    println("CoHiRF runtime")
    @btime run_cohirf_iterations($X, q=$q, R=$R, C=$C, n_init=10, max_iter=300)
    println("KMeans runtime")
    @btime kmeans_ninit($X, $C, n_init=10)
end