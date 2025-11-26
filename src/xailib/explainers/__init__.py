"""
Explainer implementations for XAI-Lib.

This subpackage contains implementations of various explanation methods
for different data types (tabular, image, text, time series).

Tabular Data Explainers
-----------------------

- :class:`~xailib.explainers.lime_explainer.LimeXAITabularExplainer`:
  LIME (Local Interpretable Model-agnostic Explanations) for tabular data.
- :class:`~xailib.explainers.shap_explainer_tab.ShapXAITabularExplainer`:
  SHAP (SHapley Additive exPlanations) for tabular data.
- :class:`~xailib.explainers.lore_explainer.LoreTabularExplainer`:
  LORE (LOcal Rule-based Explanations) for tabular data.

Image Data Explainers
---------------------

- :class:`~xailib.explainers.lime_explainer.LimeXAIImageExplainer`:
  LIME for image data.
- :class:`~xailib.explainers.gradcam_explainer.GradCAMImageExplainer`:
  GradCAM (Gradient-weighted Class Activation Mapping) for images.
- :class:`~xailib.explainers.gradcam_explainer.GradCAMPlusPlusImageExplainer`:
  GradCAM++ for improved image explanations.
- :class:`~xailib.explainers.rise_explainer.RiseXAIImageExplainer`:
  RISE (Randomized Input Sampling for Explanation) for images.
- :class:`~xailib.explainers.intgrad_explainer.IntgradImageExplainer`:
  Integrated Gradients for image explanations.
- :class:`~xailib.explainers.abele_explainer`:
  ABELE (Adversarial Black-box Explainer) for images.

Text Data Explainers
--------------------

- :class:`~xailib.explainers.lime_explainer.LimeXAITextExplainer`:
  LIME for text data.

Time Series Explainers
----------------------

- :class:`~xailib.explainers.lasts_explainer`:
  LASTS for time series explanations.

Transparent-by-Design Models
----------------------------

- :class:`~xailib.explainers.nam_explainer_tab`:
  Neural Additive Models for interpretable predictions.

Example
-------
>>> from xailib.explainers.lime_explainer import LimeXAITabularExplainer
>>> from xailib.models.sklearn_classifier_wrapper import sklearn_classifier_wrapper
>>>
>>> bb = sklearn_classifier_wrapper(trained_model)
>>> explainer = LimeXAITabularExplainer(bb)
>>> explainer.fit(df, 'target', config={})
>>> explanation = explainer.explain(instance)
"""
