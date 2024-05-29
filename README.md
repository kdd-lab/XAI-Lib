

# XAI-Lib

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
    - still work-in-progre

## Getting Started

The library can be installed using pip or by cloning the repository.

### Using PIP

The library can be installed using pip by running the following command:
```bash
pip install XAI-Library
```

The command will install the library and all the dependencies.

### Using the Repository
The library can be installed by cloning the repository. We suggest to install the library within a virtual environment.
```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/kdd-lab/XAI-Lib.git
python setup.py install
```


Note
====

This project has been set up using PyScaffold 4.2.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
