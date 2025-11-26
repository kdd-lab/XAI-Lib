"""
Base classes for XAI-Lib explainability framework.

This module defines the abstract base classes that serve as the foundation
for all explainers and explanations in the XAI-Lib library. These classes
provide a unified interface for implementing various explanation methods
across different data types (tabular, image, text, time series).

Classes:
    Explainer: Abstract base class for all explainer implementations.
    Explanation: Abstract base class for all explanation representations.

Example:
    Creating a custom explainer::

        from xailib.xailib_base import Explainer, Explanation

        class MyExplanation(Explanation):
            def getFeaturesImportance(self):
                return self.feature_weights

        class MyExplainer(Explainer):
            def fit(self, X, y):
                # Train the explainer
                pass

            def explain(self, x):
                # Generate explanation for instance x
                return MyExplanation()
"""

from abc import ABC, abstractmethod


class Explainer(ABC):
    """
    Abstract base class for all explainer implementations.

    This class defines the interface that all explainers must implement.
    An explainer is responsible for generating explanations for predictions
    made by black-box machine learning models.

    The typical workflow involves:
        1. Initialize the explainer with a black-box model
        2. Call :meth:`fit` to prepare the explainer with training data
        3. Call :meth:`explain` to generate explanations for specific instances

    Attributes:
        None defined at the base level. Subclasses should define their own.

    See Also:
        :class:`Explanation`: The corresponding base class for explanations.
        :class:`xailib.xailib_tabular.TabularExplainer`: Explainer for tabular data.
        :class:`xailib.xailib_image.ImageExplainer`: Explainer for image data.
    """

    def __init__(self):
        """Initialize the Explainer base class."""
        pass

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the explainer to the training data.

        This method prepares the explainer by learning from the training data.
        The specific behavior depends on the explanation method being used.

        Args:
            X: Training features. The format depends on the data type:
                - For tabular data: pandas DataFrame or numpy array of shape (n_samples, n_features)
                - For image data: numpy array of images
                - For text data: list of strings or text documents
            y: Training labels or target values. numpy array of shape (n_samples,)

        Returns:
            None. The explainer is fitted in-place.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        pass

    @abstractmethod
    def explain(self, x):
        """
        Generate an explanation for a single instance.

        This method creates an explanation for why the black-box model
        made a specific prediction for the given instance.

        Args:
            x: The instance to explain. The format depends on the data type:
                - For tabular data: 1D numpy array or pandas Series
                - For image data: numpy array representing an image
                - For text data: string or text document

        Returns:
            Explanation: An Explanation object containing the explanation details.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        pass


class Explanation(ABC):
    """
    Abstract base class for all explanation representations.

    This class defines the interface for accessing different aspects of
    an explanation. Explanations can provide various types of information
    including feature importance, exemplars, rules, and counterfactuals.

    Different explanation methods may only support a subset of these
    information types. Methods that are not supported by a particular
    explanation type should return None.

    Attributes:
        None defined at the base level. Subclasses should define their own.

    See Also:
        :class:`Explainer`: The corresponding base class for explainers.
        :class:`xailib.xailib_tabular.TabularExplanation`: Explanation for tabular data.
        :class:`xailib.xailib_image.ImageExplanation`: Explanation for image data.
    """

    def __init__(self):
        """Initialize the Explanation base class."""
        pass

    @abstractmethod
    def getFeaturesImportance(self):
        """
        Get the feature importance values from the explanation.

        Feature importance indicates how much each feature contributed
        to the model's prediction for the explained instance.

        Returns:
            Feature importance values. The format depends on the explanation method:
                - List of tuples: [(feature_name, importance_value), ...]
                - numpy array: Array of importance values
                - None: If feature importance is not available for this explanation type

        Example:
            >>> explanation = explainer.explain(instance)
            >>> importance = explanation.getFeaturesImportance()
            >>> for feature, value in importance:
            ...     print(f"{feature}: {value:.4f}")
        """
        pass

    @abstractmethod
    def getExemplars(self):
        """
        Get exemplar instances from the explanation.

        Exemplars are instances from the training data that are similar
        to the explained instance and received the same prediction.

        Returns:
            Exemplar instances, or None if not available for this explanation type.
            The format depends on the specific explanation method.
        """
        pass

    @abstractmethod
    def getCounterExemplars(self):
        """
        Get counter-exemplar instances from the explanation.

        Counter-exemplars are instances that are similar to the explained
        instance but received a different prediction.

        Returns:
            Counter-exemplar instances, or None if not available for this explanation type.
            The format depends on the specific explanation method.
        """
        pass

    @abstractmethod
    def getRules(self):
        """
        Get the decision rules from the explanation.

        Rules are logical conditions that describe why the model made
        its prediction for the explained instance.

        Returns:
            Decision rules as a list or dictionary, or None if not available.
            For rule-based explanations like LORE, this returns the rule
            that led to the prediction.

        Example:
            >>> rules = explanation.getRules()
            >>> print(rules)
            {'premise': [{'att': 'age', 'op': '>', 'thr': 30}], 'cons': 'approved'}
        """
        pass

    @abstractmethod
    def getCounterfactualRules(self):
        """
        Get counterfactual rules from the explanation.

        Counterfactual rules describe what minimal changes to the input
        would result in a different prediction.

        Returns:
            Counterfactual rules as a list or dictionary, or None if not available.
            Each counterfactual rule describes conditions that would lead
            to a different outcome.

        Example:
            >>> cf_rules = explanation.getCounterfactualRules()
            >>> for rule in cf_rules:
            ...     print(f"To get {rule['cons']}: {rule['premise']}")
        """
        pass
