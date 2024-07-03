
|

=======
XAI-Lib
=======


An integrated library for explanation methods.

A python library to develop explaninable machine learning models. The library provides an integrated interface to setup and execute explanation methods for black boxes.

The library is designed to be modular and extensible. It provides a simple interface to add new explanation methods and integrate them with the existing ones.
The starting list of explanation methods includes:
 - For tabular data:
    - SHAP
    - LIME
    - Anchors
    - LORE
 - For image data:
    - GradCAM
    - LIME
    - SHAP
    - ABELE
 - For text data:
    - still work-in-progress
 - For time series data:
    - still work-in-progress


Getting Started
===============
The library can be installed using pip or by cloning the repository.

To install the library using pip, run the following command:

```
pip install XAI-Library
```

To have access to the latest version of the library, clone the repository and create a virtual environment. Then install the library using the following command:

```
virtualenv venv
source venv/bin/activate
pip install -e .
```


