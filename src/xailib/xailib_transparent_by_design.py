"""
Transparent-by-design model classes for XAI-Lib.

This module provides base classes for inherently interpretable models
that are transparent by design. Unlike post-hoc explanation methods,
these models provide built-in interpretability without requiring
additional explanation techniques.

Examples of transparent-by-design models include:
    - Neural Additive Models (NAM)
    - Generalized Additive Models (GAM)
    - Decision trees and rule-based models
    - Linear models with interpretable features

Classes:
    Explainer: Base class for transparent models (extended with predict methods).
    Explanation: Base class for transparent model explanations.

Example:
    Using a transparent-by-design model::

        from xailib.explainers.nam_explainer_tab import NAMExplainer

        # Create and fit the transparent model
        explainer = NAMExplainer()
        explainer.fit(X_train, y_train)

        # Get predictions with built-in explanations
        prediction = explainer.predict(X_test)
        explanation = explainer.explain(X_test[0])

Note:
    Models in this module can both make predictions AND provide
    explanations, unlike post-hoc explainers that only explain
    existing black-box models.
"""

from abc import ABC, abstractmethod


class Explainer(ABC):
    """
    Abstract base class for transparent-by-design models.

    This class extends the standard explainer interface with prediction
    methods, allowing transparent models to serve as both predictors
    and explainers. Unlike post-hoc explanation methods, transparent
    models provide inherent interpretability.

    The workflow for transparent models:
        1. Initialize the model
        2. Call :meth:`fit` to train the model
        3. Call :meth:`predict` or :meth:`predict_proba` for predictions
        4. Call :meth:`explain` for interpretable explanations

    Attributes:
        Defined by subclasses. Common attributes include model parameters
        and learned feature contributions.

    See Also:
        :class:`xailib.xailib_base.Explainer`: Base explainer for post-hoc methods.
        :mod:`xailib.explainers.nam_explainer_tab`: NAM implementation.
    """

    def __init__(self):
        """Initialize the transparent model base class."""
        pass

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the transparent model to training data.

        Args:
            X: Training features as a numpy array or pandas DataFrame.
            y: Training labels or target values.

        Returns:
            None. The model is fitted in-place.
        """
        pass

    @abstractmethod
    def explain(self, x):
        """
        Generate an explanation for an instance.

        For transparent models, explanations are derived directly from
        the model's internal structure (e.g., feature contributions).

        Args:
            x: Instance to explain.

        Returns:
            Explanation: An explanation object with interpretable information.
        """
        pass

    @abstractmethod
    def predict(self, x):
        """
        Make predictions for input instances.

        Args:
            x: Input features for prediction.

        Returns:
            Predicted class labels or regression values.
        """
        pass

    @abstractmethod
    def predict_proba(self, x):
        """
        Get prediction probabilities for input instances.

        Args:
            x: Input features for prediction.

        Returns:
            Predicted class probabilities as a numpy array
            of shape (n_samples, n_classes).
        """
        pass


class Explanation(ABC):
    """
    Abstract base class for transparent model explanations.

    This class provides the standard interface for accessing explanation
    information from transparent-by-design models. Explanations from
    transparent models are typically more detailed and accurate than
    post-hoc explanations.

    Attributes:
        Defined by subclasses. Common attributes include feature
        contributions and model-specific interpretable components.
    """

    def __init__(self):
        """Initialize the explanation base class."""
        pass

    @abstractmethod
    def getFeaturesImportance(self):
        """
        Get feature importance values from the transparent model.

        For transparent models, feature importance is typically
        derived directly from the model's learned parameters.

        Returns:
            Feature importance values.
        """
        pass

    @abstractmethod
    def getExemplars(self):
        """
        Get exemplar instances.

        Returns:
            Exemplar instances, or None if not supported.
        """
        pass

    @abstractmethod
    def getCounterExemplars(self):
        """
        Get counter-exemplar instances.

        Returns:
            Counter-exemplar instances, or None if not supported.
        """
        pass

    @abstractmethod
    def getRules(self):
        """
        Get decision rules from the transparent model.

        For rule-based transparent models, this returns the
        learned decision rules.

        Returns:
            Decision rules, or None if not applicable.
        """
        pass

    @abstractmethod
    def getCounterfactualRules(self):
        """
        Get counterfactual rules.

        Returns:
            Counterfactual rules, or None if not supported.
        """
        pass
