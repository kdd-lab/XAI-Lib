"""
PyTorch classifier wrapper for XAI-Lib.

This module provides a wrapper class for PyTorch classifiers,
allowing them to be used with XAI-Lib explainers.

Classes:
    pytorch_classifier_wrapper: Wrapper for PyTorch classifiers.

Example:
    >>> import torch.nn as nn
    >>> from xailib.models.pytorch_classifier_wrapper import pytorch_classifier_wrapper
    >>>
    >>> # Build and train your PyTorch model
    >>> model = YourPyTorchModel()
    >>> # ... training code ...
    >>>
    >>> # Wrap it for use with XAI-Lib
    >>> bb = pytorch_classifier_wrapper(model, device="cuda", n_features=10)
    >>>
    >>> # Now use with any explainer
    >>> from xailib.explainers.intgrad_explainer import IntgradImageExplainer
    >>> explainer = IntgradImageExplainer(bb)
"""

from xailib.models.bbox import AbstractBBox
import torch
import numpy as np


class pytorch_classifier_wrapper(AbstractBBox):
    """
    Wrapper class for PyTorch classifiers.

    This class wraps PyTorch models to provide the standard interface
    expected by XAI-Lib explainers. It handles device management and
    input tensor conversion automatically.

    Args:
        classifier: A trained PyTorch model (nn.Module).
        device (str, optional): Device to use for inference, either "cpu"
            or "cuda". Defaults to "cpu".
        n_features (int, optional): Number of features for reshaping the
            input tensor. If set, inputs are reshaped to (-1, n_features).
            Defaults to 1.

    Attributes:
        bbox: The wrapped PyTorch model.
        device (str): The device being used for inference.
        n_features (int): Number of features for input reshaping.

    Example:
        >>> model = MyPyTorchClassifier()
        >>> model.load_state_dict(torch.load('model.pt'))
        >>> model.eval()
        >>> wrapper = pytorch_classifier_wrapper(model, device="cuda")
        >>> predictions = wrapper.predict(X_test)
    """

    def __init__(self, classifier, device="cpu", n_features=1):
        """
        Initialize the PyTorch classifier wrapper.

        Args:
            classifier: A trained PyTorch model.
            device (str, optional): Device for inference ("cpu" or "cuda").
            n_features (int, optional): Number of features for input reshaping.
        """
        super().__init__()
        self.bbox = classifier
        self.device = device
        self.n_features = n_features

    def model(self):
        """
        Get the underlying PyTorch model.

        Returns:
            The wrapped PyTorch model (nn.Module).
        """
        return self.bbox

    def prepare_input(self, X):
        """
        Convert input data to a PyTorch tensor suitable for model inference.

        This method handles numpy array to tensor conversion, reshaping based
        on n_features, and moving the tensor to the appropriate device.

        Args:
            X: Input data, either a numpy array or a PyTorch tensor.

        Returns:
            torch.Tensor: The input data as a float tensor, reshaped to
                (-1, n_features) if n_features is set, and moved to the
                specified device (CPU or CUDA).
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)

        X = X.float()
        if self.n_features is not None:
            X = X.reshape(-1, self.n_features)

        X = X.to(self.device)

        return X

    def predict(self, X):
        """
        Make class predictions for input instances.

        For multi-class classification, returns the argmax of the output.
        For binary classification (single output), applies a 0.5 threshold.

        Args:
            X: Input features as a numpy array or PyTorch tensor.

        Returns:
            numpy.ndarray: Predicted class labels as integers.
        """
        X = self.prepare_input(X)

        with torch.no_grad():
            y = self.bbox(X)
            if y.shape[1] > 1:
                y = torch.argmax(y, dim=-1)
            else:
                y = y > 0.5

            y = y.cpu().int().numpy().flatten()

            return y

    def predict_proba(self, X):
        """
        Get prediction probabilities for input instances.

        Args:
            X: Input features as a numpy array or PyTorch tensor.

        Returns:
            numpy.ndarray: Raw model output (logits or probabilities,
                depending on the model architecture).

        Note:
            If your model outputs logits, you may need to apply softmax
            to get proper probabilities.
        """
        X = self.prepare_input(X)

        with torch.no_grad():
            y = self.bbox(X)
            y = y.cpu().numpy()

            return y