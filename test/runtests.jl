using Test
using Distances
using OrderedCollections
using Random

import CoHiRF: 
    align_labels!, 
    build_consensus_matrix!,
    find_consensus_groups!,
    propagate_labels!,
    compute_medoid, 
    assign_medoids!,
    run_cohirf_iterations

@testset "align_labels!" begin
    @testset "basic reordering" begin
        labels = [2, 2, 1, 3, 2, 1, 6, 8, 5, 8]
        align_labels!(labels)
        @test labels == [1, 1, 2, 3, 1, 2, 4, 5, 6, 5]
    end
end

@testset "build_consensus_matrix!" begin
    @testset "basic correctness" begin
        R = 10
        C = 3
        q = 5
        n = 50
        n_curr = n ÷ 2
        n_features = 200
        X_j = randn(n_features, n_curr)
        P_buffer = zeros(Int, R, n)
        P = build_consensus_matrix!(P_buffer, X_j, R=R, C=C, q=q)
        @test size(P) == (10, 25)
    end
end

@testset "find_consensus_groups!" begin
    @testset "basic correctness" begin
        cluster_membership = OrderedDict{Tuple, Vector{Int}}()
        P = [
            1 2 1 2 1
            1 2 1 2 2
        ]
        find_consensus_groups!(cluster_membership, P)
        @test Set(keys(cluster_membership)) == Set([(1,1), (1,2), (2,2)])
        @test cluster_membership[(1,1)] == [1, 3]
        @test cluster_membership[(2,2)] == [2, 4]
        @test cluster_membership[(1,2)] == [5]
    end
end

@testset "propagate_labels!" begin
    @testset "basic correctness" begin
        label_seq = [1,2,3,4,5,6,7]
        cluster_membership = OrderedDict(
            1 => [2,3],
            2 => [1],
            3 => [4,5]
        )
        children = [[1],[2],[3],[4],[5,6,7]]
        expected_label_seq = [2,1,1,3,3,3,3]
        propagate_labels!(label_seq, cluster_membership, children)
        @test expected_label_seq == label_seq
    end
end

@testset "compute_medoid!" begin
    @testset "basic correctness" begin
        X = randn(2, 500)
        d = Euclidean()
        medoid = compute_medoid(X, d)
        @test 1 <= medoid <= size(X, 2)
    end
end

@testset "assign_medoids!" begin
    @testset "basic correctness" begin
        children = Dict(1 => [1], 2 => [2,3,4], 3 => [5], 4 => [6,7,8])
        X_j = [
            1 2 3 5
            3 3 4 7]
        cluster_reps = [1,3,5,7]
        label_seq = [1,2,2,2,2,2,2,2]
        cluster_membership = OrderedDict((1,1) => [1], (2,2) => [2,3,4])
        metric = CosineDist()

        new_X_j, new_cluster_reps = assign_medoids!(
            children, 
            X_j, 
            size(X_j, 2),
            label_seq, 
            cluster_reps, 
            cluster_membership,
            metric
        )

        @test new_X_j == [
            1 5 
            3 7]
        @test new_cluster_reps == [1, 7]
        @test children[1] == [1]
        @test children[2] == [2,3,4,5,6,7,8]
    end
end

@testset "run_cohirf_iterations" begin
    @testset "basic correctness" begin
        n_features = 100
        n_clusters = 5
        samples_per_cluster = 1000
        Δ = 100
        X = reduce(hcat, (randn(n_features, samples_per_cluster) .+ Δ .* rand(n_features) for _ in 1:n_clusters))
        results = run_cohirf_iterations(X, R=10, C=3, q=10, max_iter=300)

        @test results.n_clusters == n_clusters
    end
end