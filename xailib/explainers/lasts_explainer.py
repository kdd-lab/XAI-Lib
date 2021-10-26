from xailib.xailib_ts import TSExplainer, TSExplanation
from xailib.models.bbox import AbstractBBox
from externals.late.late.explainers import lasts
from externals.late.late.neighgen import neighborhood_generators
import pandas as pd
import json
from IPython.display import HTML


class LastsTSExplanation(TSExplanation):
    def __init__(self, lasts_exp):
        super().__init__()
        self.exp = lasts_exp

    def getFeaturesImportance(self):
        return None

    def getExemplars(self):
        return self.exp['Zplus']

    def getCounterExemplars(self):
        return self.exp['Zminus']

    def getRules(self):
        return None

    def getCounterfactualRules(self):
        return None

class LastsExplainer(TSExplainer):


    def __init__(self, bb: AbstractBBox):
        super().__init__()
        self.bb = bb

    def fit(self, config):
        #qui passiamo i parametri da inizializzare, tra cui encoder, decoder, vicinato etc
        #passiamo anche i parametri per generare il vicinato
        self.neighborhood_generator = neighborhood_generators.NeighborhoodGenerator(self.bb, config.get('decoder'))
        self.expl = lasts.Lasts(self.bb, encoder = config.get('encoder'), decoder = config.get('decoder'), neighborhood_generator= self.neighborhood_generator, labels=config.get('labels',None))
        self.config = config




    def explain(self, x, z_fixed=None):
        #qui passiamo la x e la z_fixed che serve per l'encoder
        explanation = self.expl.generate_neighborhood(x, z_fixed, **self.config)
        return LastsTSExplanation(explanation)
