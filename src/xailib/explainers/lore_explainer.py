from lore_sa.encoder_decoder import ColumnTransformerEnc
from lore_sa.lore import Lore
from lore_sa.neighgen import GeneticGenerator

from xailib.xailib_tabular import TabularExplainer, TabularExplanation
from xailib.models.bbox import AbstractBBox
import pandas as pd
from lore_explainer.datamanager import prepare_dataset
from lore_explainer.lorem import LOREM

from lore_sa.dataset import TabularDataset
from lore_sa.neighgen.random import RandomGenerator
from lore_sa.surrogate import DecisionTreeSurrogate
from lore_sa.explanation import ExplanationEncoder
import json
from IPython.display import HTML


class LoreTabularExplanation(TabularExplanation):
    def __init__(self, lore_exp):
        super().__init__()
        self.exp = lore_exp
        self.expDict = None
        
    def getFeaturesImportance(self):
        return None

    def getExemplars(self):
        return None

    def getCounterExemplars(self):
        return None

    def getRules(self):
        return self.exp['rule']

    def getCounterfactualRules(self):
        return self.exp['crules']

    def plotRules(self):
        htmlStyle=HTML("""
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

        htmlPrediction=HTML(
            '''
            <h3>Why the predicted value for class <span class='red'>%s</span> is <span class='red'>%s</span> ?</h3>
            '''%(self.expDict['rule']['class_name'],self.expDict['rule']['cons'])
        )

        htmlExplanation=HTML('''
            <p>Because all the following conditions happen:</p>
            ''')


        rulesSpans=""
        for el in self.expDict['rule']['premise']:
            rulesSpans+="<span class='rule'>"+el['att'].replace("_"," ")+ " <strong>" + el['op']+ "</strong> "+ str("%.2f" %el['thr'])+"</span>"

        htmlRules=HTML("<p class='rules'>%s</p>"%(rulesSpans))

        display(htmlStyle)
        display(htmlPrediction)
        display(htmlExplanation)
        display(htmlRules)

    def plotCounterfactualRules(self):
        htmlStyle=HTML("""
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

        htmlTitleCRules=HTML('''
            <h3>The predicted value for class <span class='red'>%s</span> is <span class='red'>%s</span>.</h3>
            <h3>It would have been:</h3>
            '''%(self.expDict['rule']['class_name'], self.expDict['bb_pred'])
                        )

        display(htmlTitleCRules)
        
        cRulesDiv=''
        for idx,el in enumerate(self.expDict['crules']):
            cRulesTitle= el['cons']
            cRulesSpans=""
            for p in el['premise']:
                cRulesSpans+="<span class='crule'>"+p['att'].replace("_"," ")+ " " + p['op']+ " "+ str("%.2f" %p['thr'])+"</span>"

                
            display(HTML('''
                <div class='crules'>
                    <div>
                        <h4><span class='red'>%s</span> if the following condition holds</h4>
                    </br>%s
                    </div>
                </div>
            '''%(cRulesTitle,cRulesSpans))   
            )

class LegacyLoreTabularExplainer(TabularExplainer):
    lore_explainer = None
    random_state = 0
    bb = None  # The Black Box to be explained

    def __init__(self, bb: AbstractBBox):
        super().__init__()
        self.bb = bb

    def fit(self, _df: pd.DataFrame, class_name, config):
        df, feature_names, class_values, numeric_columns, \
        rdf, real_feature_names, features_map = prepare_dataset(_df, class_name)
        neigh_type = config['neigh_type'] if 'neigh_type' in config else 'geneticp'
        size = config['size'] if 'size' in config else 1000
        ocr = config['ocr'] if 'ocr' in config else 0.1
        ngen = config['ngen'] if 'ngen' in config else 10
        self.lore_explainer = LOREM(rdf[real_feature_names].values, self.bb.predict, feature_names, class_name, class_values,
                               numeric_columns, features_map, neigh_type=neigh_type, categorical_use_prob=True,
                               continuous_fun_estimation=False, size=size, ocr=ocr, random_state=self.random_state, ngen=ngen,
                               bb_predict_proba=self.bb.predict_proba, verbose=False)



    def explain(self, x):
        exp = self.lore_explainer.explain_instance(x, samples=1000, use_weights=True)
        return LoreTabularExplanation(exp)


class LoreTabularExplainer(TabularExplainer):
    lore_explainer = None
    random_state = 0
    bb = None  # The Black Box to be explained

    def __init__(self, bb: AbstractBBox):
        super().__init__()
        self.bb = bb

    def fit(self, _df: pd.DataFrame, class_name, config):
        dataset = TabularDataset(_df, class_name)
        dataset.df.dropna(inplace=True)
        dataset.update_descriptor()

        neigh_type = config['neigh_type'] if 'neigh_type' in config else 'geneticp'
        size = config['size'] if 'size' in config else 1000
        ocr = config['ocr'] if 'ocr' in config else 0.1
        ngen = config['ngen'] if 'ngen' in config else 10

        enc = ColumnTransformerEnc(dataset.descriptor)
        generator = GeneticGenerator(self.bb, dataset, enc) if neigh_type == 'geneticp' else RandomGenerator(self.bb, dataset, enc)
        surrogate = DecisionTreeSurrogate()

        self.lore_explainer = Lore(self.bb, dataset, enc, generator, surrogate)


    def explain(self, x):
        exp = self.lore_explainer.explain(x)
        return LoreTabularExplanation(exp)