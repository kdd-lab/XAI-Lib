"""
Keras time series classifier wrapper for XAI-Lib.

This module provides a wrapper class for Keras time series classifiers,
handling the specific input/output requirements of time series models.

Classes:
    keras_classifier_wrapper: Wrapper for Keras time series classifiers.

Example:
    >>> from tensorflow import keras
    >>> from xailib.models.keras_ts_classifier_wrapper import keras_classifier_wrapper
    >>>
    >>> # Build and train your Keras time series model
    >>> model = keras.Sequential([
    ...     keras.layers.LSTM(64, input_shape=(timesteps, features)),
    ...     keras.layers.Dense(num_classes, activation='softmax')
    ... ])
    >>> model.fit(X_train, y_train)
    >>>
    >>> # Wrap it for use with XAI-Lib
    >>> bb = keras_classifier_wrapper(model)
"""

from xailib.models.bbox import AbstractBBox
import numpy as np


class keras_classifier_wrapper(AbstractBBox):
    """
    Wrapper class for Keras time series classifiers.

    This class wraps Keras models designed for time series classification.
    It handles the conversion of model outputs to class predictions and
    probability estimates.

    Args:
        classifier: A trained Keras time series model.

    Attributes:
        bbox: The wrapped Keras model.

    Note:
        For time series models, inputs typically have shape
        (n_samples, n_timesteps, n_features).
    """

    def __init__(self, classifier):
        """
        Initialize the Keras time series classifier wrapper.

        Args:
            classifier: A trained Keras model for time series classification.
        """
        super().__init__()
        self.bbox = classifier

    def model(self):
        """
        Get the underlying Keras model.

        Returns:
            The wrapped Keras model object.
        """
        return self.bbox

    def predict(self, X):
        """
        Make class predictions for time series input instances.

        For multi-class classification, returns the argmax of the output.

        Args:
            X: Input time series as a numpy array with 3 dimensions
                (n_samples, n_timesteps, n_features).

        Returns:
            numpy.ndarray: Predicted class labels as a 1D array.
        """
        y = self.bbox.predict(X)
        # For multi-class output, get the argmax
        if len(y.shape) > 1 and (y.shape[1] != 1):
            y = np.argmax(y, axis=1)
        return y.ravel()

    def predict_proba(self, X):
        """
        Get prediction probabilities for time series input instances.

        Args:
            X: Input time series as a numpy array with 3 dimensions
                (n_samples, n_timesteps, n_features).

        Returns:
            numpy.ndarray: Predicted class probabilities.

        Note:
            Keras predict() already returns probabilities for softmax outputs.
        """
        return self.bbox.predict(X)
