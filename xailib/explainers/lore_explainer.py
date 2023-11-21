from xailib.xailib_tabular import TabularExplainer, TabularExplanation
from xailib.models.bbox import AbstractBBox
import pandas as pd
import numpy as np

from lore_sa.dataset import TabularDataset
from lore_sa.neighgen.random import RandomGenerator
from lore_sa.surrogate import DecisionTreeSurrogate

import json
from IPython.display import HTML


class LoreTabularExplanation(TabularExplanation):
    def __init__(self, bbox: AbstractBBox):
        """
        LOREM Explainer for tabular data.

        Parameters
        ----------
        bbox [AbstractBBox]:
        """
        super().__init__()
        self.exp = lore_exp

    def getFeaturesImportance(self):
        return None

    def getExemplars(self):
        return None

    def getCounterExemplars(self):
        return None

    def getRules(self):
        return self.rule

    def getCounterfactualRules(self):
        return self.crules

    def plotRules(self):
        htmlStyle = HTML("""
                <style>
                .red {
                background-color:firebrick;
                padding:3px 5px 3px 5px;
                border-radius:5px;

                color:white;
                }
                .rules{
                    margin-top:10px;
                    font-weight: 400;
                }
                .rule{
                padding:5px 20px 5px 20px;
                border-radius:5px;
                margin-right:5px;
                font-size:12px;
                line-height:20px;
                display: block;
                margin-bottom: 10px;
                width: fit-content;

                color:white;
                background-color:firebrick;
                opacity:0.8;
                }
                </style>
                """
                         )

        htmlPrediction = HTML(
            '''
            <h3>Why the predicted value for class <span class='red'>%s</span> is <span class='red'>%s</span> ?</h3>
            ''' % (self.rule['class_name'], self.rule['cons'])
        )

        htmlExplanation = HTML('''
            <p>Because all the following conditions happen:</p>
            ''')

        rulesSpans = ""
        for el in self.rule['premise']:
            rulesSpans += "<span class='rule'>" + el['att'].replace("_", " ") + " <strong>" + el[
                'op'] + "</strong> " + str("%.2f" % el['thr']) + "</span>"

        htmlRules = HTML("<p class='rules'>%s</p>" % (rulesSpans))

        display(htmlStyle)
        display(htmlPrediction)
        display(htmlExplanation)
        display(htmlRules)

    def plotCounterfactualRules(self):
        htmlStyle = HTML("""
                <style>
                .red {
                background-color:firebrick;
                padding:3px 5px 3px 5px;
                border-radius:5px;
                color:white;
                }
                .crules{
                    margin-top:10px;
                    font-weight: 400;
                }
                .crule{            
                padding:5px 20px 5px 20px;
                border-radius:5px;
                margin-right:5px;
                font-size:12px;
                line-height:20px;
                display: block;
                margin-bottom: 10px;
                width: fit-content;

                color:#202020;
                background-color:gold;
                }
                </style>
                """
                         )
        display(htmlStyle)

        htmlTitleCRules = HTML('''
            <h3>The predicted value for class <span class='red'>%s</span> is <span class='red'>%s</span>.</h3>
            <h3>It would have been:</h3>
            ''' % (self.crules['class_name'])
                               )

        display(htmlTitleCRules)

        cRulesDiv = ''
        for idx, el in enumerate(self.crules):
            cRulesTitle = el['cons']
            cRulesSpans = ""
            for p in el['premise']:
                cRulesSpans += "<span class='crule'>" + p['att'].replace("_", " ") + " " + p['op'] + " " + str(
                    "%.2f" % p['thr']) + "</span>"

            display(HTML('''
                <div class='crules'>
                    <div>
                        <h4><span class='red'>%s</span> if the following condition holds</h4>
                    </br>%s
                    </div>
                </div>
            ''' % (cRulesTitle, cRulesSpans))
                    )


class LoreTabularExplainer(TabularExplainer):
    lore_explainer = None
    random_state = 0
    bb = None  # The Black Box to be explained

    def __init__(self, bbox: AbstractBBox):
        """
        LOREM Explainer for tabular data.
        Parameters
        ----------
        bbox [AbstractBBox]:
        """
        super().__init__()
        self.bbox = bbox

    def fit(self, df: pd.DataFrame, class_name, config):
        """

        Parameters
        ----------
        df [DataFrame]: tabular dataset
        class_name [str]: column that contains the observed class
        config [dict]: configuration dictionary with the following keys:


        Returns
        -------

        """
        self.class_name = class_name
        self.dataset = TabularDataset(data=df, class_name=self.class_name)
        self.dataset.df.dropna(inplace=True)
        self.config = config

        # encode dataset
        self.encoder = TabularEnc(self.dataset.descriptor)
        self.encoded = []
        for x in self.dataset.df.iloc:
            self.encoded.append(self.encoder.encode(x.values))

        # random generation
        self.features = [c for c in self.encoded.columns if c != self.dataset.class_name]

    def explain(self, x: np.array):
        gen = RandomGenerator()
        self.neighbour = gen.generate(x, 10000, self.dataset.descriptor, onehotencoder=self.encoder)

        # neighbour classification
        self.neighbour.df[self.class_name] = self.bbox.predict(self.neighbour.df[self.features])
        self.neighbour.set_class_name(self.class_name)

        # surrogate
        self.surrogate = DecisionTreeSurrogate()
        self.surrogate.train(self.neighbour.df[self.features].values, self.neighbour.df['class'])

        self.rule = self.surrogate.get_rule(x, self.neighbour, self.encoder)
        self.crules, self.deltas = self.surrogate.get_counterfactual_rules(x=x, class_name=self.class_name,
                                                                           feature_names=self.features,
                                                                           neighborhood_dataset=self.neighbour,
                                                                           encoder=self.encoder)
