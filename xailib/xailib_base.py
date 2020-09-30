
from abc import ABC, abstractmethod


class Explainer(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def explain(self, b, x):
        pass


class Explanation(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def getFeaturesImportance(self):
        pass

    @abstractmethod
    def getExemplars(self):
        pass

    @abstractmethod
    def getCounterExemplars(self):
        pass

    @abstractmethod
    def getRules(self):
        pass

    @abstractmethod
    def getCounterfactualRules(self):
        pass
