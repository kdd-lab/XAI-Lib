"""
XAI-Lib: An Integrated Python Library for Explainable AI.

**XAI-Lib** provides a unified interface for various explanation methods,
making machine learning models more interpretable and transparent. The library
simplifies the process of explaining black-box models across different data types.

This project is part of the `XAI Project <https://xai-project.eu/>`_ - a European
initiative focused on advancing explainable artificial intelligence research
and applications.

Main Modules
------------

Core Classes
~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    xailib.xailib_base
    xailib.xailib_tabular
    xailib.xailib_image
    xailib.xailib_text
    xailib.xailib_ts
    xailib.xailib_transparent_by_design

Explainers
~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    xailib.explainers.lime_explainer
    xailib.explainers.shap_explainer_tab
    xailib.explainers.lore_explainer
    xailib.explainers.gradcam_explainer
    xailib.explainers.rise_explainer
    xailib.explainers.intgrad_explainer
    xailib.explainers.abele_explainer
    xailib.explainers.lasts_explainer
    xailib.explainers.nam_explainer_tab

Model Wrappers
~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    xailib.models.bbox
    xailib.models.sklearn_classifier_wrapper
    xailib.models.keras_classifier_wrapper
    xailib.models.pytorch_classifier_wrapper

Data Loaders
~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    xailib.data_loaders.dataframe_loader

Metrics
~~~~~~~

.. autosummary::
    :toctree: _autosummary

    xailib.metrics.insertiondeletion

Quick Start
-----------

Here's a simple example using LIME for tabular data explanation::

    from xailib.explainers.lime_explainer import LimeXAITabularExplainer
    from xailib.models.sklearn_classifier_wrapper import sklearn_classifier_wrapper

    # Wrap your scikit-learn model
    bb = sklearn_classifier_wrapper(your_sklearn_model)

    # Create and fit the explainer
    explainer = LimeXAITabularExplainer(bb)
    explainer.fit(df, 'target_column', config={})

    # Generate explanation for an instance
    explanation = explainer.explain(instance)

    # Visualize feature importance
    explanation.plot_features_importance()

For more examples, see the `examples/ <https://github.com/kdd-lab/XAI-Lib/tree/main/examples>`_
directory in the repository.

License
-------

This project is licensed under the MIT License.

Acknowledgments
---------------

This library is developed as part of the **XAI Project** (https://xai-project.eu/),
a European initiative dedicated to advancing explainable artificial intelligence.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("XAI-Library")
except PackageNotFoundError:
    __version__ = "unknown"

# Import main classes for convenient access
from xailib.xailib_base import Explainer, Explanation

__all__ = [
    "__version__",
    "Explainer",
    "Explanation",
]
