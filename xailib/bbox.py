from abc import ABC, abstractmethod

class AbstractBBox(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass