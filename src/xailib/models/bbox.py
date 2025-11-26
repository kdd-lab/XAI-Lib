"""
Abstract base class for black-box model wrappers.

This module defines the :class:`AbstractBBox` interface that all model
wrappers must implement. Model wrappers provide a unified API for
interacting with machine learning models from different frameworks
(scikit-learn, Keras, PyTorch, etc.).

Classes:
    AbstractBBox: Abstract base class for black-box model wrappers.

Example:
    Creating a custom model wrapper::

        from xailib.models.bbox import AbstractBBox

        class MyModelWrapper(AbstractBBox):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def predict(self, X):
                return self.model.predict(X)

            def predict_proba(self, X):
                return self.model.predict_proba(X)

See Also:
    :class:`xailib.models.sklearn_classifier_wrapper.sklearn_classifier_wrapper`
    :class:`xailib.models.keras_classifier_wrapper.KerasClassifierWrapper`
    :class:`xailib.models.pytorch_classifier_wrapper.PytorchClassifierWrapper`
"""

from abc import ABC, abstractmethod


class AbstractBBox(ABC):
    """
    Abstract base class for black-box model wrappers.

    This class defines the interface that all model wrappers must implement.
    Wrappers provide a consistent API for explainers to interact with
    different types of machine learning models.

    All explainers in XAI-Lib expect models to be wrapped using a class
    that inherits from AbstractBBox.

    Attributes:
        Defined by subclasses. Common attributes include the wrapped model
        and any preprocessing functions.

    Example:
        >>> from xailib.models.sklearn_classifier_wrapper import sklearn_classifier_wrapper
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> rf = RandomForestClassifier()
        >>> rf.fit(X_train, y_train)
        >>> bb = sklearn_classifier_wrapper(rf)
        >>> predictions = bb.predict(X_test)
    """

    def __init__(self):
        """Initialize the AbstractBBox base class."""
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions for input instances.

        Args:
            X: Input features. The format depends on the model type:
                - For tabular models: numpy array of shape (n_samples, n_features)
                - For image models: numpy array of images
                - For text models: list of strings

        Returns:
            numpy.ndarray: Predicted class labels for classification,
                or predicted values for regression.
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Get prediction probabilities for input instances.

        Args:
            X: Input features in the same format as :meth:`predict`.

        Returns:
            numpy.ndarray: Predicted class probabilities of shape
                (n_samples, n_classes) for classification models.

        Raises:
            NotImplementedError: If the underlying model does not
                support probability predictions.
        """
        pass