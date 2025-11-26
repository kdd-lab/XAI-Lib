"""
Scikit-learn time series classifier wrapper for XAI-Lib.

This module provides a wrapper class for scikit-learn classifiers
designed for time series data, handling the specific input shape
requirements of time series models.

Classes:
    sklearn_classifier_wrapper: Wrapper for scikit-learn time series classifiers.

Example:
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from xailib.models.sklearn_ts_classifier_wrapper import sklearn_classifier_wrapper
    >>>
    >>> # Train your scikit-learn model on flattened time series
    >>> rf = RandomForestClassifier()
    >>> rf.fit(X_train_2d, y_train)
    >>>
    >>> # Wrap it for use with XAI-Lib (handles 3D to 2D conversion)
    >>> bb = sklearn_classifier_wrapper(rf)
"""

from xailib.models.bbox import AbstractBBox


class sklearn_classifier_wrapper(AbstractBBox):
    """
    Wrapper class for scikit-learn time series classifiers.

    This class wraps scikit-learn classifiers for time series data,
    handling the conversion from 3D time series format to the 2D
    format expected by scikit-learn.

    Args:
        classifier: A trained scikit-learn classifier.

    Attributes:
        bbox: The wrapped scikit-learn classifier.

    Note:
        This wrapper converts 3D time series input (n_samples, n_timesteps, n_features)
        to 2D format (n_samples, n_timesteps) by taking the first feature dimension.
    """

    def __init__(self, classifier):
        """
        Initialize the scikit-learn time series classifier wrapper.

        Args:
            classifier: A trained scikit-learn classifier.
        """
        super().__init__()
        self.bbox = classifier

    def model(self):
        """
        Get the underlying scikit-learn model.

        Returns:
            The wrapped scikit-learn classifier object.
        """
        return self.bbox

    def predict(self, X):
        """
        Make class predictions for time series input instances.

        Converts 3D time series input to 2D format before prediction.

        Args:
            X: Input time series as a numpy array with 3 dimensions
                (n_samples, n_timesteps, n_features).

        Returns:
            numpy.ndarray: Predicted class labels as a 1D array.
        """
        # Convert from 3D to 2D by taking first feature dimension
        X = X[:, :, 0]
        return self.bbox.predict(X).ravel()

    def predict_proba(self, X):
        """
        Get prediction probabilities for time series input instances.

        Converts 3D time series input to 2D format before prediction.

        Args:
            X: Input time series as a numpy array with 3 dimensions
                (n_samples, n_timesteps, n_features).

        Returns:
            numpy.ndarray: Predicted class probabilities.
        """
        # Convert from 3D to 2D by taking first feature dimension
        X = X[:, :, 0]
        return self.bbox.predict_proba(X)
