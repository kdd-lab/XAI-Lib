from xailib.models.bbox import AbstractBBox
import torch
import numpy as np

class pytorch_classifier_wrapper(AbstractBBox):
    def __init__(self, classifier, device = "cpu", n_features = None):
        super().__init__()
        self.bbox = classifier
        self.device = device
        self.n_features = n_features

    def model(self):
        return self.bbox

    def prepare_input(self, X):
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