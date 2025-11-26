"""
ABELE (Adversarial Black-box Explainer generating Latent Exemplars) for image data.

This module provides the ABELE explainer for image classification models.
ABELE generates explanations by learning a latent representation of images
and using it to find exemplars, counter-exemplars, and generate rules.

ABELE is particularly useful for:
    - Finding prototypical examples that explain a prediction
    - Generating counterfactual examples showing what would change the prediction
    - Extracting human-readable rules from image classifications

Classes:
    ABELEImageExplanation: Explanation class for ABELE.
    ABELEImageExplainer: Explainer class implementing ABELE algorithm.

References:
    Guidotti, R., Monreale, A., Matwin, S., & Pedreschi, D. (2019).
    Black Box Explanation by Learning Image Exemplars in the Latent Feature Space.
    ECML PKDD.

Example:
    >>> from xailib.explainers.abele_explainer import ABELEImageExplainer
    >>> from xailib.models.keras_classifier_wrapper import keras_classifier_wrapper
    >>>
    >>> bb = keras_classifier_wrapper(trained_model)
    >>> explainer = ABELEImageExplainer(bb)
    >>> explainer.fit(config={'...': '...'})
    >>> explanation = explainer.explain(image)
    >>> prototypes = explanation.getExemplars(num_prototypes=5)
"""

from xailib.xailib_image import ImageExplainer, ImageExplanation
from xailib.models.bbox import AbstractBBox
from externals.ABELE.ilore.ilorem import ILOREM
from externals.ABELE.ilore.util import neuclidean


class ABELEImageExplanation(ImageExplanation):
    """
    Explanation class for ABELE image explanations.

    This class wraps the ABELE explanation result and provides methods
    to access different aspects of the explanation: rules, exemplars,
    counter-exemplars, and feature importance.

    Args:
        abele_exp: The raw ABELE explanation object from ILOREM.

    Attributes:
        exp: The underlying ABELE explanation object.
    """

    def __init__(self, abele_exp):
        """
        Initialize the ABELE explanation.

        Args:
            abele_exp: Raw explanation from ILOREM explain_instance.
        """
        super().__init__()
        self.exp = abele_exp

    def getFeaturesImportance(self, features=None, samples=400):
        """
        Get the image-based rule showing important features.

        Args:
            features: Optional feature specification for the rule.
            samples (int): Number of samples to use for rule extraction.
                Defaults to 400.

        Returns:
            Image rule highlighting important features for the prediction.
        """
        return self.exp.get_image_rule(features=features, samples=samples)

    def getExemplars(self, num_prototypes):
        """
        Get prototype images that support the prediction.

        Prototypes are images that satisfy the decision rule and
        received the same prediction as the query image.

        Args:
            num_prototypes (int): Number of prototype images to return.

        Returns:
            List of prototype images respecting the decision rule.
        """
        return self.exp.get_prototypes_respecting_rule(num_prototypes=num_prototypes)

    def getCounterExemplars(self):
        """
        Get counterfactual prototype images.

        Counterfactual prototypes are images that would receive a
        different prediction from the query image.

        Returns:
            List of counterfactual prototype images.
        """
        return self.exp.get_counterfactual_prototypes()

    def getRules(self):
        """
        Get the decision rule as a human-readable string.

        Returns:
            str: String representation of the decision rule.
        """
        return self.exp.rstr()

    def getCounterfactualRules(self):
        """
        Get counterfactual rules as a human-readable string.

        Returns:
            str: String representation of the counterfactual rules.
        """
        return self.exp.cstr()


class ABELEImageExplainer(ImageExplainer):
    """
    ABELE (Adversarial Black-box Explainer generating Latent Exemplars) explainer.

    This explainer uses the ABELE algorithm to generate explanations for
    image classification models by learning a latent representation and
    extracting decision rules, exemplars, and counterfactuals.

    Args:
        bb (AbstractBBox): The black-box model wrapper to explain.

    Attributes:
        bb: The black-box model wrapper.
        exp: The ILOREM explainer instance (after fitting).

    Example:
        >>> explainer = ABELEImageExplainer(model_wrapper)
        >>> explainer.fit(config={
        ...     'autoencoder': autoencoder,
        ...     'latent_dim': 128,
        ...     # ... other ILOREM parameters
        ... })
        >>> explanation = explainer.explain(query_image)
    """

    def __init__(self, bb: AbstractBBox):
        """
        Initialize the ABELE explainer.

        Args:
            bb (AbstractBBox): Black-box model wrapper to explain.
        """
        super().__init__()
        self.bb = bb

    def fit(self, config):
        """
        Configure the ABELE explainer with the given parameters.

        Args:
            config (dict): Configuration dictionary with ILOREM parameters.
                See ILOREM documentation for available options.

        Returns:
            None. The explainer is configured in-place.
        """
        self.exp = ILOREM(**config)

    def explain(self, img, num_samples=300, use_weights=True, metric=neuclidean):
        """
        Generate an ABELE explanation for an image.

        Args:
            img: Query image to explain.
            num_samples (int): Number of samples for neighborhood generation.
                Defaults to 300.
            use_weights (bool): Whether to use weighted sampling.
                Defaults to True.
            metric: Distance metric for similarity computation.
                Defaults to neuclidean (normalized Euclidean).

        Returns:
            ABELEImageExplanation: Explanation object with access to rules,
                exemplars, and counterfactuals.
        """
        return ABELEImageExplanation(
            self.exp.explain_instance(
                img, num_samples=num_samples, use_weights=use_weights, metric=metric
            )
        )