using CondaPkg
CondaPkg.add(["numpy", "pandas", "openml"])
using PythonCall
using CategoricalArrays

# Add current path for dataset_preprocessor
sys = pyimport("sys")
sys.path.append(Py(@__DIR__))

pyexec("""
    import sys
    import os
    import pandas as pd
    import numpy as np

    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    from dataset_preprocessor import load_and_preprocess_openml_dataset
    """,
    Main)

function load_openml_dataset(dataset_id::Int; standardize::Bool=false)
    result = pyeval(
        "load_and_preprocess_openml_dataset(dataset_id, standardize)", 
        Main, 
        (
            dataset_id=dataset_id, 
            standardize=standardize
        )
    )

    julia_result = Dict(
        "X" => pyconvert(Array{Float64}, result["X"].values),
        "y" => categorical(pyconvert(Array{Union{String, Int}}, result["y"].values)),
        "dataset_name" => pyconvert(String, result["dataset_name"]),
        "n_samples" => pyconvert(Int, result["n_samples"]),
        "n_features" => pyconvert(Int, result["n_features"]),
        "n_classes" => pyconvert(Int, result["n_classes"]),
        "cat_features_names" => pyconvert(Array{String}, result["cat_features_names"]),
        "cont_features_names" => pyconvert(Array{String}, result["cont_features_names"]),
        "cat_dims" => pyconvert(Array{Int}, result["cat_dims"]),
        "preprocessing_info" => pyconvert(Dict, result["preprocessing_info"])
    )

    println(" Dataset: $(julia_result["dataset_name"])")
    println(" Shape: $(julia_result["n_samples"]) × $(julia_result["n_features"])")

    return julia_result
end

# Test function
function test_basic_functionality()
    println("Testing basic functionality...")
    dataset_id = 61  # Iris dataset
    result = load_openml_dataset(dataset_id, standardize=false)
    println(" Final shape: $(size(result["X"]))")
    println(" First few target values: $(result["y"][1:3])")
    return result
end

# Run tests if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    test_basic_functionality()
end