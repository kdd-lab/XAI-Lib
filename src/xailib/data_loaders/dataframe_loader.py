"""
DataFrame loading and preparation utilities for XAI-Lib.

This module provides utilities for loading and preparing pandas DataFrames
for use with XAI-Lib tabular data explainers.

Functions:
    prepare_dataframe: Prepare a DataFrame for use with explainers.

Example:
    >>> from xailib.data_loaders.dataframe_loader import prepare_dataframe
    >>>
    >>> df, feature_names, class_values, numeric_columns, \\
    ...     rdf, real_feature_names, features_map = prepare_dataframe(df, 'target')
"""

from lore_explainer.datamanager import prepare_dataset


def prepare_dataframe(df, class_field):
    """
    Prepare a pandas DataFrame for use with XAI-Lib explainers.

    This function wraps the LORE library's prepare_dataset function to
    process a DataFrame for use with various XAI-Lib explainers. It
    handles feature encoding, categorical variable mapping, and other
    preprocessing steps.

    Args:
        df (pd.DataFrame): Input DataFrame containing features and target.
        class_field (str): Name of the target/class column in the DataFrame.

    Returns:
        tuple: A tuple containing:
            - df: Processed DataFrame
            - feature_names: List of feature column names
            - class_values: List of unique class values
            - numeric_columns: List of numeric column names
            - rdf: Reconstructed DataFrame with original encoding
            - real_feature_names: Original feature names before encoding
            - features_map: Mapping of encoded to original features

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, 35],
        ...     'income': [50000, 60000, 70000],
        ...     'approved': [0, 1, 1]
        ... })
        >>> result = prepare_dataframe(df, 'approved')
        >>> df_processed, feature_names, class_values, *rest = result
    """
    return prepare_dataset(df, class_field)
