from xailib.xailib_tabular import TabularExplainer
from xailib.models.bbox import AbstractBBox
import pandas as pd
from externals.LORE.datamanager import prepare_dataset
from externals.LORE.lore.lorem import LOREM

class LoreTabularExplainer(TabularExplainer):
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
        return exp
