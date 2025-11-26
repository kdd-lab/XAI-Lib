"""
Image data explainability classes for XAI-Lib.

This module provides base classes for explaining predictions on image data.
It extends the base :class:`~xailib.xailib_base.Explainer` and
:class:`~xailib.xailib_base.Explanation` classes with image-specific
functionality.

Image explanations typically highlight which regions of an image
contributed most to a model's prediction, using techniques such as:
    - Saliency maps and heatmaps
    - Superpixel importance
    - Activation visualizations

Classes:
    ImageExplainer: Base class for image data explainers.
    ImageExplanation: Base class for image data explanations.

Example:
    Using GradCAM for image explanation::

        from xailib.explainers.gradcam_explainer import GradCAMImageExplainer
        from xailib.models.pytorch_classifier_wrapper import pytorch_classifier_wrapper

        # Wrap your model
        bb = pytorch_classifier_wrapper(your_pytorch_model)

        # Create and fit explainer
        explainer = GradCAMImageExplainer(bb)
        explainer.fit(target_layers=[model.layer4])

        # Generate explanation
        heatmap = explainer.explain(image, class_index)

See Also:
    :mod:`xailib.explainers.gradcam_explainer`: GradCAM implementation for image data.
    :mod:`xailib.explainers.lime_explainer`: LIME implementation for image data.
    :mod:`xailib.explainers.rise_explainer`: RISE implementation for image data.
    :mod:`xailib.explainers.intgrad_explainer`: Integrated Gradients for image data.
"""

from abc import abstractmethod
from xailib.xailib_base import Explainer, Explanation


class ImageExplainer(Explainer):
    """
    Abstract base class for image data explainers.

    This class extends the base :class:`~xailib.xailib_base.Explainer` class
    with functionality specific to image data. Image explainers work with
    numpy arrays representing images and provide visual explanations
    (typically heatmaps or saliency maps) for model predictions.

    Subclasses implement specific explanation methods such as GradCAM,
    LIME, RISE, or Integrated Gradients for image data.

    Attributes:
        Defined by subclasses. Common attributes include the black-box model
        wrapper and target layers for gradient-based methods.

    See Also:
        :class:`xailib.explainers.gradcam_explainer.GradCAMImageExplainer`: GradCAM implementation.
        :class:`xailib.explainers.lime_explainer.LimeXAIImageExplainer`: LIME implementation.
        :class:`xailib.explainers.rise_explainer.RiseXAIImageExplainer`: RISE implementation.
    """

    def __init__(self):
        """Initialize the ImageExplainer base class."""
        super().__init__()

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the explainer to the image training data.

        For most image explainers, this method sets up the necessary
        components for generating explanations (e.g., target layers for
        GradCAM, mask generation for RISE).

        Args:
            X: Training images or configuration parameters.
                The exact format depends on the specific method.
            y: Training labels or additional configuration.

        Returns:
            None. The explainer is fitted in-place.
        """
        pass

    @abstractmethod
    def explain(self, b, x):
        """
        Generate an explanation for an image instance.

        Args:
            b: Black-box model or prediction function.
            x: Image to explain as a numpy array.

        Returns:
            Explanation output, typically a heatmap or saliency map
            as a numpy array with the same spatial dimensions as the input.
        """
        pass


class ImageExplanation(Explanation):
    """
    Abstract base class for image data explanations.

    This class extends the base :class:`~xailib.xailib_base.Explanation` class
    with functionality specific to image data. Image explanations typically
    contain saliency maps, heatmaps, or segmentation-based importance values.

    Note:
        Most image explainers return the explanation directly (as a numpy
        array) rather than wrapping it in an ImageExplanation object.
        This class is provided for consistency and future extensions.

    Attributes:
        Defined by subclasses. Common attributes include the saliency map
        and segment importance values.
    """

    def __init__(self):
        """Initialize the ImageExplanation base class."""
        super().__init__()

    @abstractmethod
    def getFeaturesImportance(self):
        """
        Get feature (region) importance values for the image.

        For image data, "features" typically correspond to image regions
        or superpixels.

        Returns:
            Importance values for image regions, or None if not available.
        """
        pass

    @abstractmethod
    def getExemplars(self):
        """
        Get exemplar images similar to the explained image.

        Returns:
            Exemplar images, or None if not supported.
        """
        pass

    @abstractmethod
    def getCounterExemplars(self):
        """
        Get counter-exemplar images with different predictions.

        Returns:
            Counter-exemplar images, or None if not supported.
        """
        pass

    @abstractmethod
    def getRules(self):
        """
        Get decision rules for the image prediction.

        Returns:
            Rules, or None if not supported for image explanations.
        """
        pass

    @abstractmethod
    def getCounterfactualRules(self):
        """
        Get counterfactual rules for the image prediction.

        Returns:
            Counterfactual rules, or None if not supported.
        """
        pass
