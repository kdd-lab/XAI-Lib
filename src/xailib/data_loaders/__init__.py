"""
Data loading utilities for XAI-Lib.

This subpackage provides utilities for loading and preparing data
for use with XAI-Lib explainers.

Available Loaders
-----------------

- :func:`~xailib.data_loaders.dataframe_loader.prepare_dataframe`:
  Prepare a pandas DataFrame for use with XAI-Lib explainers.

Example
-------
>>> from xailib.data_loaders.dataframe_loader import prepare_dataframe
>>>
>>> df, feature_names, class_values, numeric_columns, \\
...     rdf, real_feature_names, features_map = prepare_dataframe(df, 'target')
"""
