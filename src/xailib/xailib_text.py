"""
Text data explainability classes for XAI-Lib.

This module provides base classes for explaining predictions on text data.
It extends the base :class:`~xailib.xailib_base.Explainer` and
:class:`~xailib.xailib_base.Explanation` classes with text-specific
functionality.

Text explanations typically highlight which words or phrases contributed
most to a model's prediction. Common use cases include:
    - Sentiment analysis explanation
    - Text classification explanation
    - Named entity recognition explanation

Classes:
    TextExplainer: Base class for text data explainers.
    TextExplanation: Base class for text data explanations.

Example:
    Using LIME for text explanation::

        from xailib.explainers.lime_explainer import LimeXAITextExplainer
        from xailib.models.sklearn_classifier_wrapper import sklearn_classifier_wrapper

        # Wrap your model
        bb = sklearn_classifier_wrapper(your_text_classifier)

        # Create and fit explainer
        explainer = LimeXAITextExplainer(bb)
        explainer.fit(class_names=['negative', 'positive'])

        # Generate explanation
        explanation = explainer.explain("This movie was great!")

See Also:
    :mod:`xailib.explainers.lime_explainer`: LIME implementation for text data.

Note:
    Text explanation support is currently being expanded. Additional
    methods will be added in future releases.
"""

from abc import abstractmethod
from xailib.xailib_base import Explainer, Explanation


class TextExplainer(Explainer):
    """
    Abstract base class for text data explainers.

    This class extends the base :class:`~xailib.xailib_base.Explainer` class
    with functionality specific to text data. Text explainers work with
    strings or text documents and provide word/phrase-level importance
    scores for model predictions.

    Subclasses implement specific explanation methods such as LIME
    for text classification models.

    Attributes:
        Defined by subclasses. Common attributes include the black-box model
        wrapper and text preprocessing parameters.

    See Also:
        :class:`xailib.explainers.lime_explainer.LimeXAITextExplainer`: LIME implementation.
    """

    def __init__(self):
        """Initialize the TextExplainer base class."""
        super().__init__()

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the explainer for text data.

        For most text explainers, this method sets up the necessary
        components for generating explanations (e.g., tokenizer,
        class names).

        Args:
            X: Training texts or configuration parameters.
            y: Training labels or class names.

        Returns:
            None. The explainer is fitted in-place.
        """
        pass

    @abstractmethod
    def explain(self, b, x):
        """
        Generate an explanation for a text instance.

        Args:
            b: Black-box model or prediction function.
            x: Text to explain as a string.

        Returns:
            Explanation with word/phrase importance scores.
        """
        pass


class TextExplanation(Explanation):
    """
    Abstract base class for text data explanations.

    This class extends the base :class:`~xailib.xailib_base.Explanation` class
    with functionality specific to text data. Text explanations typically
    contain word or phrase importance scores.

    Attributes:
        Defined by subclasses. Common attributes include word importance
        scores and highlighted text segments.
    """

    def __init__(self):
        """Initialize the TextExplanation base class."""
        super().__init__()

    @abstractmethod
    def getFeaturesImportance(self):
        """
        Get word/phrase importance values for the text.

        Returns:
            Word importance scores as a list of tuples (word, importance),
            or None if not available.
        """
        pass

    @abstractmethod
    def getExemplars(self):
        """
        Get exemplar texts similar to the explained text.

        Returns:
            Exemplar texts, or None if not supported.
        """
        pass

    @abstractmethod
    def getCounterExemplars(self):
        """
        Get counter-exemplar texts with different predictions.

        Returns:
            Counter-exemplar texts, or None if not supported.
        """
        pass

    @abstractmethod
    def getRules(self):
        """
        Get decision rules for the text prediction.

        Returns:
            Rules, or None if not supported for text explanations.
        """
        pass

    @abstractmethod
    def getCounterfactualRules(self):
        """
        Get counterfactual rules for the text prediction.

        Returns:
            Counterfactual rules, or None if not supported.
        """
        pass
