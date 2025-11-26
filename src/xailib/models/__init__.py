"""
Model wrappers for XAI-Lib.

This subpackage provides wrapper classes for machine learning models from
different frameworks (scikit-learn, Keras, PyTorch). These wrappers provide
a unified interface for explainers to interact with various model types.

Available Wrappers
------------------

General Purpose:
    - :class:`~xailib.models.sklearn_classifier_wrapper.sklearn_classifier_wrapper`:
      Wrapper for scikit-learn classifiers.
    - :class:`~xailib.models.keras_classifier_wrapper.keras_classifier_wrapper`:
      Wrapper for Keras/TensorFlow classifiers.
    - :class:`~xailib.models.pytorch_classifier_wrapper.pytorch_classifier_wrapper`:
      Wrapper for PyTorch classifiers.

Time Series:
    - :class:`~xailib.models.keras_ts_classifier_wrapper.keras_classifier_wrapper`:
      Wrapper for Keras time series classifiers.
    - :class:`~xailib.models.sklearn_ts_classifier_wrapper.sklearn_classifier_wrapper`:
      Wrapper for scikit-learn time series classifiers.

Base Class:
    - :class:`~xailib.models.bbox.AbstractBBox`:
      Abstract base class for all model wrappers.

Example
-------
>>> from sklearn.ensemble import RandomForestClassifier
>>> from xailib.models.sklearn_classifier_wrapper import sklearn_classifier_wrapper
>>>
>>> rf = RandomForestClassifier().fit(X_train, y_train)
>>> bb = sklearn_classifier_wrapper(rf)
>>> predictions = bb.predict(X_test)
"""
