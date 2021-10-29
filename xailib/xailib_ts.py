from abc import abstractmethod
from xailib.xailib_base import Explainer, Explanation
import pandas as pd
import numpy as np

import altair as alt
from altair import expr
from IPython.display import HTML


class TSExplanation(Explanation):

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


class TSExplainer(Explainer):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, X, y, config):
        pass

    @abstractmethod
    def explain(self, b, x) -> TSExplanation:
        pass



