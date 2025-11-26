"""
LIME (Local Interpretable Model-agnostic Explanations) implementation for XAI-Lib.

This module provides LIME explainers for tabular, image, and text data.
LIME is a popular explanation method that approximates the behavior of
a black-box model locally using an interpretable surrogate model.

LIME works by:
    1. Generating a neighborhood of perturbed samples around the instance to explain
    2. Getting predictions for these samples from the black-box model
    3. Training an interpretable model (e.g., linear model) on the neighborhood
    4. Using the interpretable model to explain the prediction

Classes:
    LimeXAITabularExplanation: Explanation class for LIME tabular explanations.
    LimeXAITabularExplainer: LIME explainer for tabular data.
    LimeXAIImageExplainer: LIME explainer for image data.
    LimeXAITextExplainer: LIME explainer for text data.

References:
    Ribeiro, M. T., Singh, S., & Guestrin, C. (2016).
    "Why Should I Trust You?": Explaining the Predictions of Any Classifier.
    KDD 2016.

Example:
    Using LIME for tabular data::

        from xailib.explainers.lime_explainer import LimeXAITabularExplainer
        from xailib.models.sklearn_classifier_wrapper import sklearn_classifier_wrapper

        bb = sklearn_classifier_wrapper(trained_model)
        explainer = LimeXAITabularExplainer(bb)
        explainer.fit(df, 'target', config={'discretize_continuous': True})
        explanation = explainer.explain(instance)
        explanation.plot_features_importance()

See Also:
    :mod:`lime`: The underlying LIME library.
    :class:`xailib.explainers.shap_explainer_tab.ShapXAITabularExplainer`: Alternative explanation method.
"""

import pandas as pd
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np

from xailib.models.bbox import AbstractBBox
from xailib.xailib_tabular import TabularExplainer, TabularExplanation
from xailib.xailib_image import ImageExplainer
from xailib.xailib_text import TextExplainer
from lore_explainer.datamanager import prepare_dataset
from lime.lime_tabular import LimeTabularExplainer
from lime.lime_image import LimeImageExplainer
from lime.lime_text import LimeTextExplainer


class LimeXAITabularExplanation(TabularExplanation):
    """
    Explanation class for LIME tabular explanations.

    This class wraps the LIME explanation result and provides methods
    to access feature importance and visualize the explanation.

    Args:
        lime_exp: The raw LIME explanation object.

    Attributes:
        exp: The underlying LIME explanation object.

    Example:
        >>> explanation = explainer.explain(instance)
        >>> importance = explanation.getFeaturesImportance()
        >>> explanation.plot_features_importance()
    """

    def __init__(self, lime_exp):
        """
        Initialize the LIME explanation.

        Args:
            lime_exp: Raw explanation from LimeTabularExplainer.
        """
        super().__init__()
        self.exp = lime_exp

    def getFeaturesImportance(self):
        """
        Get feature importance as a list of (feature, weight) tuples.

        Returns:
            list: List of tuples (feature_description, weight) showing
                how each feature contributed to the prediction.
        """
        return self.exp.as_list()

    def getExemplars(self):
        """
        Get exemplar instances.

        Note:
            LIME does not provide exemplars.

        Returns:
            None: Not available for LIME.
        """
        return None

    def getCounterExemplars(self):
        """
        Get counter-exemplar instances.

        Note:
            LIME does not provide counter-exemplars.

        Returns:
            None: Not available for LIME.
        """
        return None

    def getRules(self):
        """
        Get decision rules.

        Note:
            LIME provides feature weights, not explicit rules.
            Use getFeaturesImportance() instead.

        Returns:
            None: Not available for LIME.
        """
        return None

    def getCounterfactualRules(self):
        """
        Get counterfactual rules.

        Note:
            LIME does not provide counterfactual rules.

        Returns:
            None: Not available for LIME.
        """
        return None

    def plot_features_importance(self, fontDimension=10):
        """
        Plot an interactive visualization of feature importance.

        Creates an Altair chart showing feature weights with an
        interactive slider to filter by importance threshold.

        Args:
            fontDimension (int, optional): Base font size for the chart.
                Defaults to 10.

        Returns:
            None. Displays the chart using IPython display.
        """
        dataToPlot = pd.DataFrame(self.exp.as_list(), columns=['name', 'value'])
        dataToPlot['value'] = dataToPlot['value'].astype('float64')

        super().plot_features_importance_from(dataToPlot, fontDimension)


class LimeXAITabularExplainer(TabularExplainer):
    """
    LIME explainer for tabular data.

    This explainer uses LIME (Local Interpretable Model-agnostic Explanations)
    to explain predictions on tabular data by training a local linear model.

    Args:
        bb (AbstractBBox): The black-box model wrapper to explain.

    Attributes:
        bb: The black-box model wrapper.
        lime_explainer: The underlying LimeTabularExplainer (after fitting).

    Example:
        >>> explainer = LimeXAITabularExplainer(model_wrapper)
        >>> explainer.fit(df, 'target', config={
        ...     'discretize_continuous': True,
        ...     'feature_selection': 'auto'
        ... })
        >>> explanation = explainer.explain(instance, num_samples=1000)
    """

    lime_explainer = None

    def __init__(self, bb: AbstractBBox):
        """
        Initialize the LIME tabular explainer.

        Args:
            bb (AbstractBBox): Black-box model wrapper to explain.
        """
        super().__init__()
        self.bb = bb

    def fit(self, _df: pd.DataFrame, class_name, config):
        """
        Fit the LIME explainer to the training data.

        Args:
            _df (pd.DataFrame): Training DataFrame with features and target.
            class_name (str): Name of the target column.
            config (dict): Configuration dictionary with optional parameters:
                - 'feature_selection': Feature selection method ('auto', 'forward', etc.)
                - 'discretize_continuous' (bool): Whether to discretize continuous features
                - 'discretizer': Discretizer type ('quartile' or 'decile')
                - 'sample_around_instance' (bool): Sampling strategy
                - 'kernel_width': Width of the kernel for weighting
                - 'kernel': Custom kernel function

        Returns:
            None. The explainer is fitted in-place.
        """
        df, feature_names, class_values, numeric_columns, \
        _, _, _ = prepare_dataset(_df, class_name)
        feature_selection = config['feature_selection'] if 'feature_selection' in config else None
        discretize_continuous = config['discretize_continuous'] if 'discretize_continuous' in config else False
        discretizer = config['discretizer'] if 'discretizer' in config else 'quartile'
        sample_around_instance = config['sample_around_instance'] if 'sample_around_instance' in config else False
        kernel_width = config['kernel_width'] if 'kernel_width' in config else None
        kernel = config['kernel'] if 'kernel' in config else None
        self.lime_explainer = LimeTabularExplainer(
            df[feature_names].values, feature_names=feature_names,
            class_names=class_values, feature_selection=feature_selection,
            discretize_continuous=discretize_continuous, discretizer=discretizer,
            sample_around_instance=sample_around_instance, kernel_width=kernel_width,
            kernel=kernel
        )

    def explain(self, x, classifier_fn=None, num_samples=1000, top_labels=5):
        """
        Generate a LIME explanation for a tabular instance.

        Args:
            x: Instance to explain as a numpy array.
            classifier_fn (callable, optional): Custom prediction function.
                If None, uses the black-box model's predict_proba.
            num_samples (int): Number of samples in the neighborhood.
                Defaults to 1000.
            top_labels (int): Number of top labels to explain.
                Defaults to 5.

        Returns:
            LimeXAITabularExplanation: Explanation object with feature weights.
        """
        if classifier_fn:
            self.classifier_fn = classifier_fn
        else:
            self.classifier_fn = self.bb.predict_proba
        exp = self.lime_explainer.explain_instance(
            x,
            self.classifier_fn,
            num_samples=num_samples,
            top_labels=top_labels
        )
        return LimeXAITabularExplanation(exp)

    def plot_lime_values(self, exp, range_start, range_end):
        """
        Plot LIME feature importance values as a bar chart.

        Args:
            exp: LIME explanation (list of (feature, weight) tuples).
            range_start (int): Start index for features to display.
            range_end (int): End index for features to display.

        Returns:
            None. Displays the matplotlib figure.
        """
        feature_names = [a_tuple[0] for a_tuple in exp]
        exp = [a_tuple[1] for a_tuple in exp]
        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize=(10, 8))
        plt.bar(feature_names[range_start:range_end], exp[range_start:range_end], facecolor='lightblue', width=0.5)
        plt.xticks(feature_names[range_start:range_end], rotation='vertical')
        plt.margins(0.1)
        plt.subplots_adjust(bottom=0.25)
        plt.show()


class LimeXAIImageExplainer(ImageExplainer):
    """
    LIME explainer for image data.

    This explainer uses LIME to explain image classification predictions
    by segmenting the image and determining which segments are most
    important for the prediction.

    Args:
        bb (AbstractBBox): The black-box model wrapper to explain.

    Attributes:
        bb: The black-box model wrapper.
        lime_explainer: The underlying LimeImageExplainer (after fitting).
    """

    lime_explainer = None

    def __init__(self, bb: AbstractBBox):
        """
        Initialize the LIME image explainer.

        Args:
            bb (AbstractBBox): Black-box model wrapper to explain.
        """
        super().__init__()
        self.bb = bb

    def fit(self, verbose=False):
        """
        Initialize the LIME image explainer.

        Args:
            verbose (bool): Whether to print verbose output.
                Defaults to False.

        Returns:
            None. The explainer is initialized in-place.
        """
        self.lime_explainer = LimeImageExplainer(verbose=False)

    def explain(self, image, classifier_fn=None, segmentation_fn=None, top_labels=5, num_samples=1000):
        """
        Generate a LIME explanation for an image.

        Args:
            image: Query image to explain as a numpy array.
            classifier_fn (callable, optional): Function that takes images and
                returns predictions. If None, uses black_box.predict.
            segmentation_fn (callable, optional): Function to segment the image.
                If None, uses quickshift segmentation.
            top_labels (int): Number of top labels to explain. Defaults to 5.
            num_samples (int): Number of perturbed images to generate.
                Defaults to 1000.

        Returns:
            LIME ImageExplanation object with superpixel importance values.
        """
        if classifier_fn:
            self.classifier_fn = classifier_fn
        else:
            self.classifier_fn = self.bb.predict

        exp = self.lime_explainer.explain_instance(
            image,
            self.classifier_fn,
            segmentation_fn=segmentation_fn,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples
        )
        return exp

    def plot_lime_values(self, image, explanation, figsize=(15, 5)):
        """
        Plot a visualization of the LIME image explanation.

        Creates a three-panel figure showing:
            1. The original query image
            2. A heatmap of superpixel importance
            3. An overlay of the heatmap on the original image

        Args:
            image: The original image used in the explain function.
            explanation: LIME explanation object returned by explain().
            figsize (tuple): Figure size as (width, height). Defaults to (15, 5).

        Returns:
            None. Displays the matplotlib figure.
        """
        F, ax = plt.subplots(1, 3, figsize=figsize)
        ax[0].imshow(image)
        ax[0].axis('off')
        ax[0].set_title('Query Image')

        # plot heatmap
        ind = explanation.top_labels[0]
        dict_heatmap = dict(explanation.local_exp[ind])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
        ax[1].imshow(heatmap, cmap='coolwarm', vmin=-heatmap.max(), vmax=heatmap.max())
        ax[1].axis('off')
        ax[1].set_title('Super Pixel Heatmap Explanation')

        # plot overlap
        ax[2].imshow(image)
        ax[2].imshow(heatmap, alpha=0.5, cmap='coolwarm')
        ax[2].axis('off')
        ax[2].set_title('Overlap of Query Image and Heatmap')


class LimeXAITextExplainer(TextExplainer):
    """
    LIME explainer for text data.

    This explainer uses LIME to explain text classification predictions
    by perturbing words in the text and measuring their impact on predictions.

    Args:
        bb (AbstractBBox): The black-box model wrapper to explain.

    Attributes:
        bb: The black-box model wrapper.
        lime_explainer: The underlying LimeTextExplainer (after fitting).
    """

    lime_explainer = None

    def __init__(self, bb: AbstractBBox):
        """
        Initialize the LIME text explainer.

        Args:
            bb (AbstractBBox): Black-box model wrapper to explain.
        """
        super().__init__()
        self.bb = bb

    def fit(self, class_names=None, verbose=False):
        """
        Initialize the LIME text explainer.

        Args:
            class_names (list, optional): List of class names ordered
                according to the classifier output. If None, class names
                will be '0', '1', etc.
            verbose (bool): Whether to print verbose output.
                Defaults to False.

        Returns:
            None. The explainer is initialized in-place.
        """
        self.lime_explainer = LimeTextExplainer(class_names=class_names, verbose=False)

    def explain(self, sentence, classifier_fn=None, num_samples=1000, plot=False):
        """
        Generate a LIME explanation for a text sentence.

        Args:
            sentence (str): Query text to explain.
            classifier_fn (callable, optional): Function that takes text
                and returns predictions. If None, uses black_box.predict.
            num_samples (int): Number of perturbed texts to generate.
                Defaults to 1000.
            plot (bool): Whether to display a matplotlib plot of the
                explanation. Defaults to False.

        Returns:
            LIME TextExplanation object with word importance values.
        """
        if classifier_fn:
            self.classifier_fn = classifier_fn
        else:
            self.classifier_fn = self.bb.predict
            
        exp = self.lime_explainer.explain_instance(sentence,
                                                   self.classifier_fn,
                                                   num_samples=num_samples)
        
        if plot:
            exp.as_pyplot_figure()
        return exp
        
        
        
        
        