#!/usr/bin/env python3
"""
Standalone dataset preprocessing function for OpenML datasets.
Extracted from cohirf pipeline to be callable from external programs like Julia.
"""

import pandas as pd
import numpy as np
import openml
from typing import Dict, Tuple, Optional


def load_and_preprocess_openml_dataset(dataset_id: int, standardize: bool = False) -> Dict:
    """
    Load and preprocess an OpenML dataset following the cohirf pipeline.

    This function replicates the exact preprocessing pipeline used in the cohirf
    clustering experiments for categorical and continuous features.

    Parameters
    ----------
    dataset_id : int
        The OpenML dataset ID to load and preprocess
    standardize : bool, default=False
        Whether to standardize continuous features (z-score normalization)

    Returns
    -------
    dict
        Dictionary containing:
        - 'X': Preprocessed feature matrix (pandas DataFrame)
        - 'y': Target variable (pandas Series)
        - 'dataset_name': Name of the dataset
        - 'n_samples': Number of samples
        - 'n_features': Number of preprocessed features
        - 'n_classes': Number of unique classes in target
        - 'cat_features_names': Names of original categorical features
        - 'cont_features_names': Names of original continuous features
        - 'cat_dims': Cardinalities of categorical features (before one-hot)
        - 'preprocessing_info': Dictionary with preprocessing details

    Example
    -------
    >>> result = load_and_preprocess_openml_dataset(61, standardize=True)
    >>> X, y = result['X'], result['y']
    >>> print(f"Loaded {result['dataset_name']} with {X.shape[1]} features")
    """

    # Load dataset from OpenML
    dataset = openml.datasets.get_dataset(dataset_id)
    target = dataset.default_target_attribute
    X, y, cat_ind, att_names = dataset.get_data(target=target)

    # Identify categorical and continuous features
    cat_features_names = [att_names[i] for i, value in enumerate(cat_ind) if value is True]
    cont_features_names = [att_names[i] for i, value in enumerate(cat_ind) if value is False]

    # Store original categorical dimensions before preprocessing
    cat_dims = [len(X[cat_feature].cat.categories) for cat_feature in cat_features_names]

    # Dataset metadata
    n_classes = len(y.unique())
    dataset_name = dataset.name
    n_samples_orig = X.shape[0]
    n_features_orig = X.shape[1]

    # Store preprocessing info
    preprocessing_info = {
        'original_shape': (n_samples_orig, n_features_orig),
        'cat_features_original_count': len(cat_features_names),
        'cont_features_original_count': len(cont_features_names),
        'standardize': standardize
    }

    # Keep track of dropped features
    dropped_high_cardinality = []
    one_hot_encoded_features = []

    # --- Categorical feature preprocessing ---
    if cat_features_names:
        # Convert categorical features to codes
        for cat_feature in cat_features_names:
            X[cat_feature] = X[cat_feature].cat.codes
            X[cat_feature] = X[cat_feature].replace(-1, np.nan).astype('category')

        # Fill missing values with the most frequent value (mode)
        mode_values = X[cat_features_names].mode().iloc[0]
        X[cat_features_names] = X[cat_features_names].fillna(mode_values)

        # Calculate categorical dimensions after filling missing values
        cat_dims = [len(X[cat_feature].cat.categories) for cat_feature in cat_features_names]

        # One-hot encode categorical features with < 10 categories
        cat_features_names_less_10 = [cat_feature for cat_feature, cat_dim in zip(cat_features_names, cat_dims)
                                      if cat_dim < 10]

        if cat_features_names_less_10:
            X = pd.get_dummies(X, columns=cat_features_names_less_10, drop_first=True)
            one_hot_encoded_features = cat_features_names_less_10

        # Drop categorical features with ≥ 10 categories (high cardinality)
        cat_features_drop = [cat_feature for cat_feature in cat_features_names
                           if cat_feature not in cat_features_names_less_10]

        if cat_features_drop:
            X = X.drop(columns=cat_features_drop)
            dropped_high_cardinality = cat_features_drop

    # --- Continuous feature preprocessing ---
    if cont_features_names:
        # Fill missing values with the median
        median_values = X[cont_features_names].median()
        X[cont_features_names] = X[cont_features_names].fillna(median_values)

        # Optional standardization of continuous features
        if standardize:
            mean_values = X[cont_features_names].mean()
            std_values = X[cont_features_names].std()
            X[cont_features_names] = (X[cont_features_names] - mean_values) / std_values
            preprocessing_info['continuous_standardized'] = True
        else:
            preprocessing_info['continuous_standardized'] = False

        # Cast to float
        X[cont_features_names] = X[cont_features_names].astype(float)

    # Drop features with all NaN values (zero-variance features)
    features_before_drop = X.shape[1]
    X = X.dropna(axis=1, how='all')
    dropped_zero_variance = features_before_drop - X.shape[1]

    # Update preprocessing info
    preprocessing_info.update({
        'final_shape': X.shape,
        'one_hot_encoded_features': one_hot_encoded_features,
        'dropped_high_cardinality': dropped_high_cardinality,
        'dropped_zero_variance_count': dropped_zero_variance,
        'missing_values_imputed': True
    })

    # Convert to appropriate data types for compatibility
    X = X.astype(float)

    return {
        'X': X,
        'y': y,
        'dataset_name': dataset_name,
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': n_classes,
        'cat_features_names': cat_features_names,
        'cont_features_names': cont_features_names,
        'cat_dims': cat_dims,
        'preprocessing_info': preprocessing_info
    }