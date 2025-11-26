"""
Scikit-learn classifier wrapper for XAI-Lib.

This module provides a wrapper class for scikit-learn classifiers,
allowing them to be used with XAI-Lib explainers.

Classes:
    sklearn_classifier_wrapper: Wrapper for scikit-learn classifiers.

Example:
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from xailib.models.sklearn_classifier_wrapper import sklearn_classifier_wrapper
    >>>
    >>> # Train your model
    >>> rf = RandomForestClassifier()
    >>> rf.fit(X_train, y_train)
    >>>
    >>> # Wrap it for use with XAI-Lib
    >>> bb = sklearn_classifier_wrapper(rf)
    >>>
    >>> # Now use with any explainer
    >>> from xailib.explainers.lime_explainer import LimeXAITabularExplainer
    >>> explainer = LimeXAITabularExplainer(bb)
"""

from xailib.models.bbox import AbstractBBox


class sklearn_classifier_wrapper(AbstractBBox):
    """
    Wrapper class for scikit-learn classifiers.

    This class wraps scikit-learn compatible classifiers to provide
    the standard interface expected by XAI-Lib explainers.

    Args:
        classifier: A trained scikit-learn classifier with `predict`
            and `predict_proba` methods.

    Attributes:
        bbox: The wrapped scikit-learn classifier.

    Example:
        >>> from sklearn.ensemble import GradientBoostingClassifier
        >>> clf = GradientBoostingClassifier().fit(X_train, y_train)
        >>> wrapper = sklearn_classifier_wrapper(clf)
        >>> wrapper.predict(X_test[:5])
        array([0, 1, 1, 0, 1])
    """

    def __init__(self, classifier):
        """
        Initialize the sklearn classifier wrapper.

        Args:
            classifier: A trained scikit-learn compatible classifier.
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
        Make class predictions for input instances.

        Args:
            X: Input features as a numpy array of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted class labels of shape (n_samples,).
        """
        return self.bbox.predict(X)

    def predict_proba(self, X):
        """
        Get prediction probabilities for input instances.

        Args:
            X: Input features as a numpy array of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted class probabilities of shape
                (n_samples, n_classes).
        """
        return self.bbox.predict_proba(X)
