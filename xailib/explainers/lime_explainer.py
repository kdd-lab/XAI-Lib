import pandas as pd

from xailib.models.bbox import AbstractBBox
from xailib.xailib_tabular import TabularExplainer
from lore.datamanager import prepare_dataset
from lime.lime_tabular import LimeTabularExplainer


class LimeXAITabularExplainer(TabularExplainer):
    lime_explainer = None

    def __init__(self, bb: AbstractBBox):
        super().__init__()
        self.bb = bb

    def fit(self, _df: pd.DataFrame, class_name):
        df, feature_names, class_values, numeric_columns, \
        _, _, _ = prepare_dataset(_df, class_name)

        self.lime_explainer = LimeTabularExplainer(df[feature_names].values, feature_names=feature_names,
                                              class_names=class_values, discretize_continuous=False)


    def explain(self, x):
        exp = self.lime_explainer.explain_instance(x, self.bb.predict_proba)
        return exp