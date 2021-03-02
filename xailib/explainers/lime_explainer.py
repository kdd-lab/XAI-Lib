import pandas as pd

from xailib.models.bbox import AbstractBBox
from xailib.xailib_tabular import TabularExplainer
from xailib.xailib_image import ImageExplainer

from lore.datamanager import prepare_dataset
from lime.lime_tabular import LimeTabularExplainer
from lime.lime_image import LimeImageExplainer

class LimeXAITabularExplainer(TabularExplainer):
    lime_explainer = None

    def __init__(self, bb: AbstractBBox):
        super().__init__()
        self.bb = bb

    def fit(self, _df: pd.DataFrame, class_name):
        df, feature_names, class_values, numeric_columns, \
        _, _, _ = prepare_dataset(_df, class_name)

        self.lime_explainer = LimeTabularExplainer(df[feature_names].values, 
                                                   feature_names=feature_names,
                                                   class_names=class_values, 
                                                   discretize_continuous=False)


    def explain(self, x):
        exp = self.lime_explainer.explain_instance(x, self.bb.predict_proba)
        return exp


class LimeXAIImageExplainer(ImageExplainer):
    lime_explainer = None

    def __init__(self, bb: AbstractBBox):
        super().__init__()
        self.bb = bb

    def wrapper(self, x):
        return self.bb.predict(x/255)

    def fit(self):            
        self.lime_explainer = LimeImageExplainer(verbose = False)

    def explain(self, x, top_labels=5, num_samples=2000):
        if x.dtype != 'int':
            exp = self.lime_explainer.explain_instance((x*255).astype(int), 
                                                       self.wrapper, 
                                                       top_labels=top_labels, 
                                                       hide_color=0, 
                                                       num_samples=num_samples)
        else: 
            exp = self.lime_explainer.explain_instance(x, 
                                                       self.bb.predict, 
                                                       top_labels=top_labels, 
                                                       hide_color=0, 
                                                       num_samples=num_samples)

        return exp
