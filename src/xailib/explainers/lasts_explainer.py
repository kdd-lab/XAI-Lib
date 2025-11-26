"""
LASTS (Local Agnostic Subsequence-based Time Series) explainer for time series data.

This module provides the LASTS explainer for time series classification models.
LASTS generates explanations by identifying subsequences (shapelets) that are
important for the prediction, along with exemplar and counter-exemplar time series.

LASTS is particularly useful for:
    - Understanding which parts of a time series led to a prediction
    - Finding similar time series (exemplars) with the same prediction
    - Finding contrasting time series (counter-exemplars) with different predictions

Classes:
    LastsTSExplanation: Explanation class for LASTS.
    LastsExplainer: Explainer class implementing LASTS algorithm.

Example:
    >>> from xailib.explainers.lasts_explainer import LastsExplainer
    >>> from xailib.models.keras_ts_classifier_wrapper import keras_classifier_wrapper
    >>>
    >>> bb = keras_classifier_wrapper(trained_ts_model)
    >>> explainer = LastsExplainer(bb)
    >>> explainer.fit(config={
    ...     'encoder': encoder,
    ...     'decoder': decoder,
    ...     'labels': ['class_0', 'class_1']
    ... })
    >>> explanation = explainer.explain(time_series)
    >>> exemplars = explanation.getExemplars()
"""

from xailib.xailib_ts import TSExplainer, TSExplanation
from xailib.models.bbox import AbstractBBox
from externals.late.late.explainers import lasts
from externals.late.late.neighgen import neighborhood_generators
import pandas as pd
import json
from IPython.display import HTML


class LastsTSExplanation(TSExplanation):
    """
    Explanation class for LASTS time series explanations.

    This class wraps the LASTS explanation result and provides methods
    to access exemplar and counter-exemplar time series.

    Args:
        lasts_exp (dict): The raw LASTS explanation dictionary containing
            'Zplus' (exemplars) and 'Zminus' (counter-exemplars).

    Attributes:
        exp (dict): The underlying LASTS explanation dictionary.
    """

    def __init__(self, lasts_exp):
        """
        Initialize the LASTS explanation.

        Args:
            lasts_exp (dict): Raw explanation from LASTS explainer.
        """
        super().__init__()
        self.exp = lasts_exp

    def getFeaturesImportance(self):
        """
        Get temporal feature importance values.

        Note:
            LASTS does not provide direct feature importance.
            Use getExemplars() and getCounterExemplars() for explanation.

        Returns:
            None: Feature importance is not available for LASTS.
        """
        return None

    def getExemplars(self):
        """
        Get exemplar time series with the same prediction.

        Exemplars (Z+) are time series from the neighborhood that
        received the same prediction as the query time series.

        Returns:
            numpy.ndarray: Array of exemplar time series.
        """
        return self.exp['Zplus']

    def getCounterExemplars(self):
        """
        Get counter-exemplar time series with different predictions.

        Counter-exemplars (Z-) are time series that would receive
        a different prediction from the query time series.

        Returns:
            numpy.ndarray: Array of counter-exemplar time series.
        """
        return self.exp['Zminus']

    def getRules(self):
        """
        Get decision rules for the prediction.

        Note:
            LASTS does not provide explicit decision rules.

        Returns:
            None: Rules are not available for LASTS.
        """
        return None

    def getCounterfactualRules(self):
        """
        Get counterfactual rules.

        Note:
            LASTS does not provide explicit counterfactual rules.

        Returns:
            None: Counterfactual rules are not available for LASTS.
        """
        return None


class LastsExplainer(TSExplainer):
    """
    LASTS (Local Agnostic Subsequence-based Time Series) explainer.

    This explainer uses the LASTS algorithm to generate explanations for
    time series classification models. It works by generating a neighborhood
    in latent space using an encoder-decoder architecture.

    Args:
        bb (AbstractBBox): The black-box model wrapper to explain.

    Attributes:
        bb: The black-box model wrapper.
        neighborhood_generator: Generator for the latent neighborhood.
        expl: The LASTS explainer instance (after fitting).
        config (dict): Configuration parameters.

    Example:
        >>> explainer = LastsExplainer(model_wrapper)
        >>> explainer.fit(config={
        ...     'encoder': trained_encoder,
        ...     'decoder': trained_decoder,
        ...     'labels': ['normal', 'anomaly'],
        ...     'n_neighbors': 100
        ... })
        >>> explanation = explainer.explain(query_ts)
    """

    def __init__(self, bb: AbstractBBox):
        """
        Initialize the LASTS explainer.

        Args:
            bb (AbstractBBox): Black-box model wrapper to explain.
        """
        super().__init__()
        self.bb = bb

    def fit(self, config):
        """
        Configure the LASTS explainer with encoder, decoder, and parameters.

        Args:
            config (dict): Configuration dictionary containing:
                - 'encoder': Trained encoder model for latent space projection
                - 'decoder': Trained decoder model for reconstruction
                - 'labels' (optional): List of class label names
                - Additional neighborhood generation parameters

        Returns:
            None. The explainer is configured in-place.
        """
        self.neighborhood_generator = neighborhood_generators.NeighborhoodGenerator(
            self.bb, config.get('decoder')
        )
        self.expl = lasts.Lasts(
            self.bb,
            encoder=config.get('encoder'),
            decoder=config.get('decoder'),
            neighborhood_generator=self.neighborhood_generator,
            labels=config.get('labels', None)
        )
        self.config = config

    def explain(self, x, z_fixed=None):
        """
        Generate a LASTS explanation for a time series.

        Args:
            x: Query time series to explain.
            z_fixed: Optional fixed latent representation for the encoder.
                If None, the encoder will compute it from x.

        Returns:
            LastsTSExplanation: Explanation object with access to exemplars
                and counter-exemplars.
        """
        explanation = self.expl.generate_neighborhood(x, z_fixed, **self.config)
        return LastsTSExplanation(explanation)
