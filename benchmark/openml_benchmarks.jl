using Clustering
using DataFrames
using Distances
using CategoricalArrays
using StatsBase
using MLJ
using CSV
using Random
using BenchmarkTools
using ProgressMeter
using CoHiRF
using Base.Threads

include("dataset_preprocessor.jl")

# =========== Benchmark OpenML Datasets ===============

const RE = r"(?<key>\w+)\s*=\s*(?<val>-?\d+)"
extract_params(s) = Dict(m[:key] => parse(Int, m[:val]) for m in eachmatch(RE, s))

dataset_df = CSV.read(joinpath(@__DIR__, "cohirf_realdata.csv"), DataFrame)
DATASETS = dataset_df[:, [:openml_id, :dataset]] |> unique

for DATASET_NAME in DATASETS.dataset
    begin 
        filtered = filter(row -> row.dataset == DATASET_NAME, dataset_df)
        @show dataset_id = filtered[1, :openml_id]

        # Load data
        full_dataset = load_openml_dataset(dataset_id; standardize=true)
        X = full_dataset["X"]'
        y = levelcode.(full_dataset["y"])

        if DATASET_NAME == "SHUTTLE"
            params = [(R, C, q) for R in 6:8 for C in 8:10 for q in 2:4]
        else
            params = [(R, C, q) for R in 2:10 for C in 2:10 for q in 2:30]
        end

        results = Vector{Tuple{Float64, Int, Int, Int}}(undef, length(params))
        p = Progress(length(params); desc="Grid search: ")

        @threads for i in eachindex(params)
            R, C, q = params[i]
            cohirf_results = run_cohirf_iterations(X, R=R, C=C, q=q, max_iter=300, n_init=10, metric=CosineDist())
            cohirf_pred = cohirf_results.labels
            cohirf_ari = randindex(y, cohirf_pred)[1]
            results[i] = (cohirf_ari, R, C, q)
            next!(p)
        end

        best_idx = argmax(first.(results))
        best_cohirf_ari, best_R, best_C, best_q = results[best_idx]
        println("Best ARI: $best_cohirf_ari (R=$best_R, C=$best_C, q=$best_q)")

        # BENCHMARKING
        # Get parameters from paper
        p = filter(row -> row.model == "COHIRF", filtered).parameters[1] |> extract_params
        R_paper = p["R"]
        C_paper = p["C"]
        q_paper = p["q"]
        @btime run_cohirf_iterations($X, R=$R_paper, C=$C_paper, q=$q_paper, max_iter=300, n_init=10, metric=CosineDist())

        println("")
    end
end