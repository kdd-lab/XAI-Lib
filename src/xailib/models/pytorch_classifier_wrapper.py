from xailib.models.bbox import AbstractBBox
import torch
import numpy as np

class pytorch_classifier_wrapper(AbstractBBox):
    def __init__(self, classifier, device = "cpu"):
        super().__init__()
        self.bbox = classifier
        self.device = device

    def model(self):
        return self.bbox

    def prepare_input(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        X = X.to(self.device)

        return X

    def predict(self, X):
        X = self.prepare_input(X)

        with torch.no_grad():
            y = self.bbox(X)
            y = torch.argmax(y, dim=-1)

            y = y.cpu().numpy()

            return y

    def predict_proba(self, X):
        X = self.prepare_input(X)

        with torch.no_grad():
            y = self.bbox(X)
            y = y.cpu().numpy()

            return y