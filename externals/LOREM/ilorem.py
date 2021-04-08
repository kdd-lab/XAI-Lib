import numpy as np

from functools import partial

from scipy.spatial.distance import cdist

from skimage.filters import sobel
from skimage.color import rgb2gray, gray2rgb
from skimage.segmentation import quickshift

from externals.LOREM.explanation import ImageExplanation
from externals.LOREM.decision_tree import learn_local_decision_tree
from externals.LOREM.ineighgen import ImageLimeGenerator
from externals.LOREM.rule import get_rule, get_counterfactual_rules
from externals.LOREM.util import calculate_feature_values, neuclidean, multilabel2str, multi_dt_predict


def default_kernel(d, kernel_width):
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))


class ILOREM(object):

    def __init__(self, bb_predict, class_name, class_values, neigh_type='lime', ocr=0.1,
                 kernel_width=None, kernel=None, random_state=0, verbose=False, **kwargs):
        """
        Arguments:
            - bb_predict: Predict function of the black box, need to deal with a list of images as input and return an array of dimension(len(input),1) of the predicted class index
            - class_name: string that represent the name of the class in the rule
            - class_values: list strings with the names of the classes
            - neight_type: type of neighbourhood generated, for now 'lime' is the only one supported
            - ocr: other class values, ratio of other class from the one predicted in the neighbourhood
            - kernel: Kernel to weights the point in the nieghbourhood
            - kernel_width : Width of the kernel
            - segmentation_fn: Segmentation Function to segment the images
        """

        np.random.seed(random_state)
        self.bb_predict = bb_predict
        self.class_name = class_name
        self.class_values = class_values
        self.neigh_type = neigh_type
        self.ocr = ocr
        self.kernel_width = kernel_width
        self.kernel = kernel
        self.verbose = verbose

        self.feature_names = None
        self.numeric_columns = None
        self.features_map = None
        self.features_map_inv = None

        self.__init_neighbor_fn(kwargs)

    def explain_instance(self, img, num_samples=1000, use_weights=True, metric='cosine', hide_color=None):
        """
        Arguments:
            - img: image to explain
            - num_samples: number of samples of the neighbourhood
            - use_weights: if weights the points using distance
            - metric: metric to use for distance of the neighbourhood, supported the metric of scipy.spatial.distance.cdist
            - hide_color: color in greyscale (0 to 255) to use to hide the image, if is None, it will hide with pixel mean of the segment 
        """
        img = img if len(img.shape) == 3 else gray2rgb(img)

        if self.verbose:
            print('generating neighborhood - %s' % self.neigh_type)

        Z, Yb, class_value, segments = self.neighgen_fn(img, num_samples, hide_color)

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

        dt = learn_local_decision_tree(Z, Yb, weights, self.class_values)
        Yc = dt.predict(Z)

        fidelity = dt.score(Z, Yb, sample_weight=weights)

        if self.verbose:
            print('retrieving explanation')

        x = Z[0]
        feature_names = [v for v in range(len(x))]
        numeric_columns = feature_names
        rule = get_rule(x, dt, feature_names, self.class_name, self.class_values, numeric_columns, False)
        crules, deltas = get_counterfactual_rules(x, Yc[0], dt, Z, Yc, feature_names, self.class_name,
                                                  self.class_values, numeric_columns, self.features_map,
                                                  self.features_map_inv)

        exp = ImageExplanation(img, segments)
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
            segmentation_fn = kwargs.get('segmentation_fn')
            if segmentation_fn is None:
                segmentation_fn = lambda image : quickshift(image)
            neighgen = ImageLimeGenerator(self.bb_predict, ocr=self.ocr, segmentation_fn=segmentation_fn)
        else:
            print('unknown neighborhood generator')
            raise Exception

        self.neighgen_fn = neighgen.generate

