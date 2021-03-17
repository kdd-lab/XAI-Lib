import numpy as np
from functools import partial

# from skimage.color import gray2rgb

from scipy.spatial.distance import cdist

from ..ilore.util import neuclidean
from ..ilore.explanation import ImageExplanation
from ..ilore.ineighgen import ImageRandomAdversarialGeneratorLatent
from ..ilore.ineighgen import ImageGeneticAdversarialGeneratorLatent
from ..ilore.ineighgen import ImageRandomGeneticAdversarialGeneratorLatent
from ..ilore.ineighgen import ImageProbaGeneticAdversarialGeneratorLatent
from ..ilore.ineighgen import ImageRandomProbaGeneticAdversarialGeneratorLatent
from ..ilore.decision_tree import learn_local_decision_tree
from ..ilore.rule import get_rule, get_counterfactual_rules


def default_kernel(d, kernel_width):
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))


class ILOREM(object):

    def __init__(self, bb_predict, class_name, class_values, neigh_type='lime', ocr=0.1,
                 kernel_width=None, kernel=None, autoencoder=None, use_rgb=True, scale=False, valid_thr=0.5,
                 filter_crules=True, random_state=0, verbose=False, **kwargs):

        np.random.seed(random_state)
        self.bb_predict = bb_predict
        self.class_name = class_name
        self.class_values = class_values
        self.neigh_type = neigh_type
        self.filter_crules = self.bb_predict if filter_crules else None
        self.ocr = ocr
        self.kernel_width = kernel_width
        self.kernel = kernel
        self.verbose = verbose
        self.random_state = random_state

        self.autoencoder = autoencoder
        self.use_rgb = use_rgb
        self.scale = scale
        self.valid_thr = valid_thr

        self.feature_names = None
        self.__init_neighbor_fn(kwargs)

    def explain_instance(self, img, num_samples=1000, use_weights=True, metric='euclidean'):

        if self.verbose:
            print('generating neighborhood - %s' % self.neigh_type)

        Z, Yb, class_value = self.neighgen_fn(img, num_samples)

        if self.verbose:
            neigh_class, neigh_counts = np.unique(Yb, return_counts=True)
            neigh_class_counts = {self.class_values[k]: v for k, v in zip(neigh_class, neigh_counts)}

            print('synthetic neighborhood class counts %s' % neigh_class_counts)

        nbr_features = Z.shape[1]

        kernel_width = np.sqrt(nbr_features) * 0.75 if self.kernel_width is None else self.kernel_width
        self.kernel_width = float(kernel_width)

        kernel = default_kernel if self.kernel is None else self.kernel
        self.kernel = partial(kernel, kernel_width=self.kernel_width)

        self.feature_names = [i for i in range(nbr_features)]

        weights = None if not use_weights else self.__calculate_weights__(Z, metric)

        if self.verbose:
            print('learning local decision tree')

        dt = learn_local_decision_tree(Z, Yb, weights, self.class_values, prune_tree=True)
        Yc = dt.predict(Z)

        fidelity = dt.score(Z, Yb, sample_weight=weights)

        if self.verbose:
            print('retrieving explanation')

        x = Z[0]
        rule = get_rule(x, dt, self.feature_names, self.class_name, self.class_values, self.feature_names)
        crules, deltas = get_counterfactual_rules(x, Yc[0], dt, Z, Yc, self.feature_names, self.class_name,
                                                  self.class_values, self.feature_names, self.neighgen.autoencoder,
                                                  self.filter_crules)

        exp = ImageExplanation(img, self.neighgen.autoencoder, self.bb_predict, self.neighgen, self.use_rgb)
        exp.bb_pred = Yb[0]
        exp.dt_pred = Yc[0]
        exp.rule = rule
        exp.crules = crules
        exp.deltas = deltas
        exp.dt = dt
        exp.fidelity = fidelity
        exp.Z = Z

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

        if self.neigh_type in ['rnd']:  # random autoencoder

            self.neighgen = ImageRandomAdversarialGeneratorLatent(self.bb_predict, ocr=0.1,
                                                                  autoencoder=self.autoencoder,
                                                                  min_width=1, min_height=1, scale=self.scale,
                                                                  valid_thr=self.valid_thr)

        elif self.neigh_type in ['gnt', 'hrg', 'gntp', 'hrgp']:
            alpha1 = kwargs.get('alpha1', 0.5)
            alpha2 = kwargs.get('alpha2', 0.5)
            metric = kwargs.get('metric', neuclidean)
            ngen = kwargs.get('ngen', 10)
            mutpb = kwargs.get('mutpb', 0.5)
            cxpb = kwargs.get('cxpb', 0.7)
            tournsize = kwargs.get('tournsize', 3)
            halloffame_ratio = kwargs.get('halloffame_ratio', 0.1)

            if self.neigh_type in ['gnt']:    # genetic autoencoder latent
                self.neighgen = ImageGeneticAdversarialGeneratorLatent(self.bb_predict, ocr=0.1,
                                                                       autoencoder=self.autoencoder,
                                                                       min_width=1, min_height=1, scale=self.scale,
                                                                       valid_thr=self.valid_thr, alpha1=alpha1,
                                                                       alpha2=alpha2, metric=metric,
                                                                       ngen=ngen, mutpb=mutpb, cxpb=cxpb,
                                                                       tournsize=tournsize,
                                                                       halloffame_ratio=halloffame_ratio,
                                                                       random_seed=self.random_state,
                                                                       verbose=self.verbose)

            elif self.neigh_type in ['hrg']:     # hybryd random genetic autoencoder latent

                self.neighgen = ImageRandomGeneticAdversarialGeneratorLatent(self.bb_predict, ocr=0.1,
                                                                             autoencoder=self.autoencoder,
                                                                             min_width=1, min_height=1,
                                                                             scale=self.scale, valid_thr=self.valid_thr,
                                                                             alpha1=alpha1, alpha2=alpha2,
                                                                             metric=metric, ngen=ngen, mutpb=mutpb,
                                                                             cxpb=cxpb, tournsize=tournsize,
                                                                             halloffame_ratio=halloffame_ratio,
                                                                             random_seed=self.random_state,
                                                                             verbose=self.verbose)

            elif self.neigh_type in ['gntp']:  # probabilistic genetic autoencoder
                bb_predict_proba = kwargs.get('bb_predict_proba', None)

                self.neighgen = ImageProbaGeneticAdversarialGeneratorLatent(self.bb_predict, ocr=0.1,
                                                                            autoencoder=self.autoencoder,
                                                                            min_width=1, min_height=1, scale=self.scale,
                                                                            valid_thr=self.valid_thr,
                                                                            alpha1=alpha1, alpha2=alpha2, metric=metric,
                                                                            ngen=ngen, mutpb=mutpb, cxpb=cxpb,
                                                                            tournsize=tournsize,
                                                                            halloffame_ratio=halloffame_ratio,
                                                                            bb_predict_proba=bb_predict_proba,
                                                                            random_seed=self.random_state,
                                                                            verbose=self.verbose)

            elif self.neigh_type in ['hrgp']:  # probabilistic genetic autoencoder
                bb_predict_proba = kwargs.get('bb_predict_proba', None)

                self.neighgen = ImageRandomProbaGeneticAdversarialGeneratorLatent(self.bb_predict, ocr=0.1,
                                                                                  autoencoder=self.autoencoder,
                                                                                  min_width=1, min_height=1,
                                                                                  scale=self.scale,
                                                                                  valid_thr=self.valid_thr,
                                                                                  alpha1=alpha1, alpha2=alpha2,
                                                                                  metric=metric, ngen=ngen, mutpb=mutpb,
                                                                                  cxpb=cxpb, tournsize=tournsize,
                                                                                  halloffame_ratio=halloffame_ratio,
                                                                                  bb_predict_proba=bb_predict_proba,
                                                                                  random_seed=self.random_state,
                                                                                  verbose=self.verbose)

        else:
            print('unknown neighborhood generator')
            raise Exception

        self.neighgen_fn = self.neighgen.generate

