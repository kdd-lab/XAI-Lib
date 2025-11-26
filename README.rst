
=======
XAI-Lib
=======

.. image:: https://img.shields.io/pypi/v/XAI-Library.svg
   :target: https://pypi.org/project/XAI-Library/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/XAI-Library.svg
   :target: https://pypi.org/project/XAI-Library/
   :alt: Python versions

.. image:: https://img.shields.io/github/license/kdd-lab/XAI-Lib.svg
   :target: https://github.com/kdd-lab/XAI-Lib/blob/main/LICENSE
   :alt: License

|

**XAI-Lib** is an integrated Python library for Explainable AI (XAI) that provides a unified interface for various explanation methods. Developed to make machine learning models more interpretable and transparent, XAI-Lib simplifies the process of explaining black-box models across different data types.

This project is part of the `XAI Project <https://xai-project.eu/>`_ - a European initiative focused on advancing explainable artificial intelligence research and applications.

|

✨ Features
===========

**XAI-Lib** is designed to be **modular**, **extensible**, and **easy to use**. It provides:

- 🔌 **Unified Interface**: Simple, consistent API for multiple explanation methods
- 📊 **Multiple Data Types**: Support for tabular, image, text, and time-series data
- 🧩 **Extensible Architecture**: Easy integration of new explanation methods
- 🎯 **Model-Agnostic**: Works with any black-box machine learning model
- 📚 **Well-Documented**: Comprehensive documentation and examples

|

🛠️ Supported Explanation Methods
==================================

Tabular Data
------------

* **SHAP** - SHapley Additive exPlanations
* **LIME** - Local Interpretable Model-agnostic Explanations
* **Anchors** - High-precision model-agnostic explanations
* **LORE** - LOcal Rule-based Explanations

Image Data
----------

* **GradCAM** - Gradient-weighted Class Activation Mapping
* **LIME** - Local Interpretable Model-agnostic Explanations
* **SHAP** - SHapley Additive exPlanations
* **ABELE** - Adversarial Black-box Explainer generating Latent Exemplars

Text Data
---------

* **LIME** - Local Interpretable Model-agnostic Explanations for text classification

Time Series Data
----------------

* **LASTS** - Local Agnostic Subsequence-based Time Series explanations

|

📦 Installation
===============

Install from PyPI
-----------------

The easiest way to install **XAI-Lib** is using pip:

.. code-block:: bash

   pip install XAI-Library

Install from Source
-------------------

For the latest development version, clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/kdd-lab/XAI-Lib.git
   cd XAI-Lib
   pip install -e .

**Recommended**: Use a virtual environment to avoid dependency conflicts:

.. code-block:: bash

   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install XAI-Lib
   pip install -e .

|

🚀 Quick Start
==============

Here's a simple example of using LIME for tabular data explanation:

.. code-block:: python

   from xailib.explainers.lime_explainer import LimeXAITabularExplainer
   from xailib.models.sklearn_classifier_wrapper import sklearn_classifier_wrapper
   
   # Wrap your scikit-learn model
   bb = sklearn_classifier_wrapper(your_trained_model)
   
   # Create and fit the LIME explainer
   explainer = LimeXAITabularExplainer(bb)
   explainer.fit(df, 'target_column', config={
       'discretize_continuous': True,
       'feature_selection': 'auto'
   })
   
   # Generate explanation for an instance
   explanation = explainer.explain(instance, num_samples=1000)
   
   # Visualize feature importance
   explanation.plot_features_importance()

For more examples and detailed usage, please check the `examples/ <https://github.com/kdd-lab/XAI-Lib/tree/main/examples>`_ directory.

|

📖 Documentation
================

For detailed documentation, tutorials, and API reference, visit:

* **GitHub Pages Documentation**: `https://kdd-lab.github.io/XAI-Lib/ <https://kdd-lab.github.io/XAI-Lib/>`_
* **Read the Docs**: `https://xai-lib.readthedocs.io/ <https://xai-lib.readthedocs.io/>`_
* **GitHub Repository**: `https://github.com/kdd-lab/XAI-Lib <https://github.com/kdd-lab/XAI-Lib>`_
* **Issue Tracker**: `https://github.com/kdd-lab/XAI-Lib/issues <https://github.com/kdd-lab/XAI-Lib/issues>`_

The documentation includes:

* **Getting Started Guide**: Installation and quick start tutorials
* **API Reference**: Complete documentation of all classes and methods
* **Examples**: Practical examples for tabular, image, and text data
* **Contributing Guide**: How to contribute to the project

|

🤝 Contributing
===============

We welcome contributions! Please see our `Contributing Guide <CONTRIBUTING.rst>`_ for details on how to:

* Report bugs and request features
* Submit pull requests
* Improve documentation
* Add new explanation methods

|

👥 Authors & Contributors
==========================

See `AUTHORS.rst <AUTHORS.rst>`_ for a complete list of contributors to this project.

|

📄 License
==========

This project is licensed under the MIT License - see the `LICENSE <LICENSE.txt>`_ file for details.

|

Acknowledgments
==================

This library is developed as part of the **XAI Project** (`https://xai-project.eu/ <https://xai-project.eu/>`_), a European initiative dedicated to advancing explainable artificial intelligence.

The Xai project focuses on the urgent open challenge of how to construct meaningful explanations of opaque AI/ML systems in the context of ai based decision making, aiming at empowering individual against undesired effects of automated decision making, implementing the right of explanation, helping people make better decisions preserving (and expand) human autonomy.


For more information about the XAI Project, visit `https://xai-project.eu/ <https://xai-project.eu/>`_.

|

📧 Contact
==========

For questions and support:

* **Email**: rinzivillo@isti.cnr.it
* **Issue Tracker**: `https://github.com/kdd-lab/XAI-Lib/issues <https://github.com/kdd-lab/XAI-Lib/issues>`_



