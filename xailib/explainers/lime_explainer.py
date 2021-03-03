import pandas as pd

from xailib.models.bbox import AbstractBBox
from xailib.xailib_tabular import TabularExplainer
from externals.LORE.datamanager import prepare_dataset
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

class LimeXAITabularExplainer(TabularExplainer):
    lime_explainer = None

    def __init__(self, bb: AbstractBBox):
        super().__init__()
        self.bb = bb

    def fit(self, _df: pd.DataFrame, class_name, config):
        df, feature_names, class_values, numeric_columns, \
        _, _, _ = prepare_dataset(_df, class_name)
        feature_selection=config['feature_selection'] if 'feature_selection' in config else None
        discretize_continuous = config['discretize_continuous'] if 'discretize_continuous' in config else False
        discretizer = config['discretizer'] if 'discretizer' in config else 'quartile'
        sample_around_instance = config['sample_around_instance'] if 'sample_around_instance' in config else False
        kernel_width = config['kernel_width'] if 'kernel_width' in config else None
        kernel = config['kernel'] if 'kernel' in config else None
        self.lime_explainer = LimeTabularExplainer(df[feature_names].values, feature_names=feature_names,
                                              class_names=class_values, feature_selection= feature_selection,
                                                   discretize_continuous=discretize_continuous, discretizer=discretizer,
                                                   sample_around_instance=sample_around_instance, kernel_width=kernel_width,
                                                   kernel=kernel)


    def explain(self, x):
        exp = self.lime_explainer.explain_instance(x, self.bb.predict_proba)
        return exp

    def plot_lime_values(self, exp, range_start, range_end):
        feature_names = [a_tuple[0] for a_tuple in exp]
        exp = [a_tuple[1] for a_tuple in exp]
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
