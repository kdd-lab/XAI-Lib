import pandas as pd
from matplotlib.pyplot import figure
from xailib.models.bbox import AbstractBBox
from xailib.xailib_tabular import TabularExplainer
from externals.LORE.datamanager import prepare_dataset
import shap
shap.initjs

class NoExplainerFound(Exception):

    def __init__(self, name):
        self.message = 'Explanator not found '+name
        super().__init__(self.message)


class ShapXAITabularExplainer(TabularExplainer):
    shap_explainer = None

    def __init__(self, bb: AbstractBBox):
        super().__init__()
        self.bb = bb

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
        return exp

    '''def force_plot(self, x, x_index):
        figure(num=None, figsize=(3, 4), dpi=100)
        shap.force_plot(self.shap_explainer.expected_value[0], self.shap_explainer.shap_values(x)[0], x, matplotlib=True)
        figure.show()'''
