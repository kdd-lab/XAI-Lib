from xailib.xailib_tabular import TabularExplainer
from xailib.models.bbox import AbstractBBox
import pandas as pd
from lore.datamanager import prepare_dataset
from lore.lorem import LOREM

class LoreTabularExplainer(TabularExplainer):
    lore_explainer = None
    random_state = 0
    bb = None  # The Black Box to be explained

    def __init__(self, bb: AbstractBBox):
        super().__init__()
        self.bb = bb

    def fit(self, _df: pd.DataFrame, class_name):
        df, feature_names, class_values, numeric_columns, \
        rdf, real_feature_names, features_map = prepare_dataset(_df, class_name)

        self.lore_explainer = LOREM(rdf[real_feature_names].values, self.bb.predict, feature_names, class_name, class_values,
                               numeric_columns, features_map, neigh_type='geneticp', categorical_use_prob=True,
                               continuous_fun_estimation=False, size=300, ocr=0.1, random_state=self.random_state, ngen=10,
                               bb_predict_proba=self.bb.predict_proba, verbose=False)

        # print(feature_names)  # (name of features after expansion for discrete attributes
        # numeric_columna (only numeri columns)
        # print(real_feature_names)  # (the original set of features)
        # features_map (bho!!)


    # return expl = explainer(bb, x)
    def explain(self, x):
        exp = self.lore_explainer.explain_instance(x, samples=300, use_weights=True)
        return exp