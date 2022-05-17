
from abc import abstractmethod
from xailib.xailib_base import Explainer, Explanation


class ImageExplainer(Explainer):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def explain(self, b, x):
        pass


class ImageExplanation(Explanation):

    def __init__(self):
        super().__init__()

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
