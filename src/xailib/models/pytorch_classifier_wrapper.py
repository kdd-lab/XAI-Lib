from xailib.models.bbox import AbstractBBox
import torch
import numpy as np

class pytorch_classifier_wrapper(AbstractBBox):
    """ Wrapper for a Pytorch classifier to be used as a BBox.

    Args:
        classifier: Pytorch model.
        device: Optional, string indicating the device to use, either "cpu" or "cuda". Default is "cpu".
        n_features: Optional, integer indicating the number of features for reshaping the input. Default is 1.
    """

    def __init__(self, classifier, device = "cpu", n_features = 1):
        super().__init__()
        self.bbox = classifier
        self.device = device
        self.n_features = n_features

    def model(self):
        return self.bbox

    def prepare_input(self, X):
        r""" Converts input data to a PyTorch tensor suitable for model inference.

        Args:
            X: Input data, either a numpy array or a PyTorch tensor.

        Returns:
            torch.Tensor: The input data as a float tensor, reshaped to (-1, n_features) if n_features is set,
            and moved to the specified device (CPU or CUDA).
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)

        X = X.float()
        if self.n_features is not None:
            X = X.reshape(-1, self.n_features)

        X = X.to(self.device)

        return X

    def predict(self, X):
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
        X = self.prepare_input(X)

        with torch.no_grad():
            y = self.bbox(X)
            y = y.cpu().numpy()

            return y