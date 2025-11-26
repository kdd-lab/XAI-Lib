"""
Time series data explainability classes for XAI-Lib.

This module provides base classes for explaining predictions on time series data.
It extends the base :class:`~xailib.xailib_base.Explainer` and
:class:`~xailib.xailib_base.Explanation` classes with time series-specific
functionality.

Time series explanations typically highlight which time steps or temporal
patterns contributed most to a model's prediction. Common use cases include:
    - Anomaly detection explanation
    - Time series classification explanation
    - Forecasting explanation

Classes:
    TSExplainer: Base class for time series data explainers.
    TSExplanation: Base class for time series data explanations.

Example:
    Using LASTS for time series explanation::

        from xailib.explainers.lasts_explainer import LastsExplainer
        from xailib.models.keras_ts_classifier_wrapper import KerasTSClassifierWrapper

        # Wrap your model
        bb = KerasTSClassifierWrapper(your_ts_model)

        # Create and fit explainer
        explainer = LastsExplainer(bb)
        explainer.fit(X_train, y_train, config)

        # Generate explanation
        explanation = explainer.explain(time_series)

See Also:
    :mod:`xailib.explainers.lasts_explainer`: LASTS implementation for time series.

Note:
    Time series explanation support is currently being expanded.
    Additional methods will be added in future releases.
"""

from abc import abstractmethod
from xailib.xailib_base import Explainer, Explanation
import pandas as pd
import numpy as np

import altair as alt
from altair import expr
from IPython.display import HTML


class TSExplanation(Explanation):
    """
    Abstract base class for time series data explanations.

    This class extends the base :class:`~xailib.xailib_base.Explanation` class
    with functionality specific to time series data. Time series explanations
    typically contain temporal importance scores and pattern-based explanations.

    Attributes:
        Defined by subclasses. Common attributes include temporal importance
        scores and identified patterns.
    """

    def __init__(self):
        """Initialize the TSExplanation base class."""
        super().__init__()

    @abstractmethod
    def getFeaturesImportance(self):
        """
        Get temporal feature importance values.

        For time series data, "features" may correspond to time steps,
        temporal windows, or derived features (e.g., shapelets).

        Returns:
            Temporal importance values, or None if not available.
        """
        pass

    @abstractmethod
    def getExemplars(self):
        """
        Get exemplar time series similar to the explained series.

        Returns:
            Exemplar time series, or None if not supported.
        """
        pass

    @abstractmethod
    def getCounterExemplars(self):
        """
        Get counter-exemplar time series with different predictions.

        Returns:
            Counter-exemplar time series, or None if not supported.
        """
        pass

    @abstractmethod
    def getRules(self):
        """
        Get decision rules for the time series prediction.

        For time series, rules may describe temporal patterns or
        conditions over time windows.

        Returns:
            Rules, or None if not supported.
        """
        pass

    @abstractmethod
    def getCounterfactualRules(self):
        """
        Get counterfactual rules for the time series prediction.

        Returns:
            Counterfactual rules describing temporal changes
            that would alter the prediction, or None if not supported.
        """
        pass


class TSExplainer(Explainer):
    """
    Abstract base class for time series data explainers.

    This class extends the base :class:`~xailib.xailib_base.Explainer` class
    with functionality specific to time series data. Time series explainers
    work with sequential data and provide temporal importance explanations
    for model predictions.

    Subclasses implement specific explanation methods such as LASTS
    for time series classification models.

    Attributes:
        Defined by subclasses. Common attributes include the black-box model
        wrapper and temporal configuration parameters.

    See Also:
        :class:`xailib.explainers.lasts_explainer.LastsExplainer`: LASTS implementation.
    """

    def __init__(self):
        """Initialize the TSExplainer base class."""
        super().__init__()

    @abstractmethod
    def fit(self, X, y, config):
        """
        Fit the explainer to the time series training data.

        Args:
            X: Training time series data, typically as a numpy array
                of shape (n_samples, n_timesteps) or
                (n_samples, n_timesteps, n_features).
            y: Training labels or target values.
            config (dict): Configuration dictionary with method-specific
                parameters for the time series explainer.

        Returns:
            None. The explainer is fitted in-place.
        """
        pass

    @abstractmethod
    def explain(self, b, x) -> TSExplanation:
        """
        Generate an explanation for a time series instance.

        Args:
            b: Black-box model or prediction function.
            x: Time series to explain as a numpy array.

        Returns:
            TSExplanation: An explanation object containing temporal
                importance scores and pattern information.
        """
        pass