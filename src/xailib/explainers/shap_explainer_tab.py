import pandas as pd
import numpy as np
from matplotlib.pyplot import figure
from xailib.models.bbox import AbstractBBox
from xailib.xailib_tabular import TabularExplainer, TabularExplanation
import shap
import matplotlib.pyplot as plt
# shap.initjs



class ShapXAITabularExplanation(TabularExplanation):
    def __init__(self, shap_exp, feature_names: list):
        super().__init__()
        self.exp = shap_exp
        self.feature_names = feature_names

    def getFeaturesImportance(self):
        return self.exp

    def getExemplars(self):
        return None

    def getCounterExemplars(self):
        return None

    def getRules(self):
        return None

    def getCounterfactualRules(self):
        return None

    def plot_features_importance(self, fontDimension=10):
        #data prepraration
        feature_values = []
        if np.ndim(self.exp) == 1 :
            feature_values = self.exp
        else:
            feature_values = self.exp[0]


        arr=np.c_[np.array(self.feature_names), feature_values]
        dataToPlot=pd.DataFrame(arr,columns=['name','value'])
        dataToPlot['value'] = dataToPlot['value'].astype('float64')

        super().plot_features_importance_from(dataToPlot, fontDimension)


class NoExplainerFound(Exception):

    def __init__(self, name):
        self.message = 'Explanator not found '+name
        super().__init__(self.message)


class ShapXAITabularExplainer(TabularExplainer):
    shap_explainer = None

    def __init__(self, bb: AbstractBBox, feature_names: list):
        super().__init__()
        self.bb = bb
        self.feature_names = feature_names

    def fit(self, config):
        if config['explainer'] == 'linear':
            self.shap_explainer = shap.LinearExplainer(self.bb.model(), config['X_train'])
        elif config['explainer'] == 'tree':
            self.shap_explainer = shap.TreeExplainer(self.bb.model())
        elif config['explainer'] == 'deep':
            self.shap_explainer = shap.DeepExplainer(self.bb, config['X_train'])
        elif config['explainer'] == 'kernel':
            self.shap_explainer = shap.KernelExplainer(self.bb.predict_proba, config['X_train'])
        else:
            raise NoExplainerFound(config['explainer'])



    def explain(self, x):
        exp = self.shap_explainer.shap_values(x)
        return ShapXAITabularExplanation(exp, self.feature_names)

    def expected_value(self, val):
        if val == -1:
            return self.shap_explainer.expected_value
        else:
            return self.shap_explainer.expected_value[val]

    def plot_shap_values(self, feature_names, exp, range_start, range_end):
        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize=(10, 8))
        plt.bar(feature_names[range_start:range_end], exp[range_start:range_end], facecolor='lightblue', width=0.5)
        # You can specify a rotation for the tick labels in degrees or with keywords.
        plt.xticks(feature_names[range_start:range_end], rotation='vertical')
        # Pad margins so that markers don't get clipped by the axes
        plt.margins(0.1)
        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(bottom=0.25)
        plt.show()

    