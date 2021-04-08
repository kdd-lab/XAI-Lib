import numpy as np
from functools import partial
from scipy.spatial.distance import cdist
from externals.LOREM.explanation import TextExplanation
from externals.LOREM.decision_tree import learn_local_decision_tree
from externals.LOREM.tneighgen import TextLimeGenerator
from externals.LOREM.rule import get_rule, get_counterfactual_rules
from externals.LOREM.util import calculate_feature_values, neuclidean, multilabel2str, multi_dt_predict

def default_kernel(d, kernel_width):
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

class TLOREM(object):

    def __init__(self, K, bb_predict, class_name, class_values, neigh_type='lime', ocr=0.1,
                 bow=True, split_expression=r'\W+', kernel_width=None, kernel=None,
                 random_state=0, verbose=False, **kwargs):

        np.random.seed(random_state)
        self.bb_predict = bb_predict
        self.K = K
        self.class_name = class_name
        self.class_values = class_values
        self.neigh_type = neigh_type
        self.vocabulary = None
        self.ocr = ocr
        self.kernel_width = kernel_width
        self.kernel = kernel
        self.verbose = verbose

        self.feature_names = None
        self.numeric_columns = None
        self.features_map = None
        self.features_map_inv = None

        self.__init_neighbor_fn(kwargs)

    def explain_instance(self, text, num_samples=1000, use_weights=True, metric=neuclidean):

        if self.verbose:
            print('generating neighborhood - %s' % self.neigh_type)

        Z, Yb, class_value, indexed_text = self.neighgen_fn(text, num_samples)

        nbr_features = Z.shape[1]

        kernel_width = np.sqrt(nbr_features) * 0.75 if self.kernel_width is None else self.kernel_width
        self.kernel_width = float(kernel_width)

        kernel = default_kernel if self.kernel is None else self.kernel
        self.kernel = partial(kernel, kernel_width=self.kernel_width)

        self.feature_names = [i for i in range(nbr_features)]

        if self.verbose:
            neigh_class, neigh_counts = np.unique(Yb, return_counts=True)
            neigh_class_counts = {self.class_values[k]: v for k, v in zip(neigh_class, neigh_counts)}

            print('synthetic neighborhood class counts %s' % neigh_class_counts)

        weights = None if not use_weights else self.__calculate_weights__(Z, metric)

        if self.verbose:
            print('learning local decision tree')

        dt = learn_local_decision_tree(Z, Yb, weights, self.class_values)
        Yc = dt.predict(Z)

        fidelity = dt.score(Z, Yb, sample_weight=weights)

        if self.verbose:
            print('retrieving explanation')

        x = Z[0]
        numeric_columns = self.feature_names
        rule = get_rule(x, dt, self.feature_names, self.class_name, self.class_values, self.feature_names, False)
        crules, deltas = get_counterfactual_rules(x, Yc[0], dt, Z, Yc, self.feature_names, self.class_name,
                                                  self.class_values,  numeric_columns, self.features_map,
                                                  self.features_map_inv)

        exp = TextExplanation(text, indexed_text)
        exp.bb_pred = Yb[0]
        exp.dt_pred = Yc[0]
        exp.rule = rule
        exp.crules = crules
        exp.deltas = deltas
        exp.dt = dt
        exp.fidelity = fidelity

        return exp

    def __calculate_weights__(self, Z, metric):
        if np.max(Z) != 1 and np.min(Z) != 0:
            Zn = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
            distances = cdist(Zn, Zn[0].reshape(1, -1), metric=metric).ravel()
        else:
            distances = cdist(Z, Z[0].reshape(1, -1), metric=metric).ravel()
        weights = self.kernel(distances)
        return weights

    def __init_neighbor_fn(self, kwargs):

        if self.neigh_type in ['lime']:
            bow = kwargs.get('bow', True)
            split_expression = kwargs.get('split_expression', r'\W+')
            neighgen = TextLimeGenerator(self.bb_predict, ocr=self.ocr, bow=bow, split_expression=split_expression)
        else:
            print('unknown neighborhood generator')
            raise Exception

        self.neighgen_fn = neighgen.generate



