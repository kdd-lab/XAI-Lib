"""
Keras classifier wrapper for XAI-Lib.

This module provides a wrapper class for Keras/TensorFlow classifiers,
allowing them to be used with XAI-Lib explainers.

Classes:
    keras_classifier_wrapper: Wrapper for Keras classifiers.

Example:
    >>> from tensorflow import keras
    >>> from xailib.models.keras_classifier_wrapper import keras_classifier_wrapper
    >>>
    >>> # Build and train your Keras model
    >>> model = keras.Sequential([...])
    >>> model.fit(X_train, y_train)
    >>>
    >>> # Wrap it for use with XAI-Lib
    >>> bb = keras_classifier_wrapper(model)
    >>>
    >>> # Now use with any explainer
    >>> from xailib.explainers.gradcam_explainer import GradCAMImageExplainer
    >>> explainer = GradCAMImageExplainer(bb)
"""

from xailib.models.bbox import AbstractBBox


class keras_classifier_wrapper(AbstractBBox):
    """
    Wrapper class for Keras/TensorFlow classifiers.

    This class wraps Keras models to provide the standard interface
    expected by XAI-Lib explainers.

    Args:
        classifier: A trained Keras model.

    Attributes:
        bbox: The wrapped Keras model.

    Example:
        >>> model = keras.Sequential([...])
        >>> model.fit(X_train, y_train)
        >>> wrapper = keras_classifier_wrapper(model)
        >>> wrapper.predict(X_test[:5])
    """

    def __init__(self, classifier):
        """
        Initialize the Keras classifier wrapper.

        Args:
            classifier: A trained Keras model.
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
        Make class predictions for input instances.

        Args:
            X: Input features as a numpy array.

        Returns:
            numpy.ndarray: Predicted class labels.
        """
        return self.bbox.predict(X)

    def predict_proba(self, X):
        """
        Get prediction probabilities for input instances.

        Args:
            X: Input features as a numpy array.

        Returns:
            numpy.ndarray: Predicted class probabilities.
        """
        return self.bbox.predict_proba(X)
