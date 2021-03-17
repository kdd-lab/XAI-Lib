import sys

import datetime
import numpy as np

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

from experiments.exputil import get_dataset
from experiments.exputil import get_black_box
from experiments.exputil import store_fcpn
from experiments.exputil import calculate_plausibilities


def main():

    dataset = sys.argv[1]
    black_box = sys.argv[2]

    # dataset = 'mnist'
    # black_box = 'RF'

    nbr_experiments = 200

    if dataset not in ['mnist', 'cifar10', 'fashion']:
        print('unknown dataset %s' % dataset)
        return -1

    if black_box not in ['RF', 'AB', 'DNN']:
        print('unknown black box %s' % black_box)
        return -1

    path = './'
    path_models = path + 'models/'
    path_results = path + 'results/fcp/'
    path_neigh = './neigh/'

    black_box_filename = path_models + '%s_%s' % (dataset, black_box)
    results_filename = path_results + 'lime_p_%s_%s.json' % (dataset, black_box)
    neigh_filename = path_neigh + 'lime_%s_%s.json' % (dataset, black_box)

    _, _, X_test, Y_test, use_rgb = get_dataset(dataset)
    bb, transform = get_black_box(black_box, black_box_filename, use_rgb, return_model=True)
    bb_predict, bb_predict_proba = get_black_box(black_box, black_box_filename, use_rgb)

    lime_explainer = lime_image.LimeImageExplainer()
    segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

    for i2e in range(nbr_experiments):
        img = X_test[i2e]

        start_time = datetime.datetime.now()
        exp = lime_explainer.explain_instance(img, bb_predict_proba, top_labels=1, hide_color=0, num_samples=1000,
                                              segmentation_fn=segmenter)
        run_time = (datetime.datetime.now() - start_time).total_seconds()

        label = bb_predict(np.array([X_test[i2e]]))[0]

        bb_probs = lime_explainer.Zl[:, label]
        lr_probs = lime_explainer.lr.predict(lime_explainer.Zlr)

        fidelity = 1 - np.sum(np.abs(bb_probs - lr_probs) < 0.01) / len(bb_probs)

        img_cdist = transform(np.array([img]))
        Z_cdist = transform(lime_explainer.Z)

        if black_box == 'DNN':
            img_cdist = np.array([x.ravel() for x in img_cdist])
            Z_cdist = np.array([x.ravel() for x in Z_cdist])

        rdist = cdist(img_cdist, Z_cdist, metric='euclidean')
        compact, compact_var = float(np.mean(rdist)), float(np.std(rdist))

        sdist = pairwise_distances(lime_explainer.Zlr, np.array([lime_explainer.Zlr[0]]), metric='cosine').ravel()
        lcompact, lcompact_var = float(np.mean(sdist)), float(np.std(sdist))

        X_test_cdist = transform(X_test)
        if black_box == 'DNN':
            X_test_cdist = np.array([x.ravel() for x in X_test_cdist])

        dist = cdist(img_cdist, X_test_cdist, metric='euclidean')
        nbr_real_instances = len(X_test)
        plausibility = calculate_plausibilities(rdist, dist, nbr_real_instances)

        print(datetime.datetime.now(), '[%s/%s] %s %s - f: %.2f, c: %.2f, lc: %.2f, p: %.2f' % (
            i2e, nbr_experiments, dataset, black_box, fidelity, compact, lcompact, plausibility[-2]))

        Z = lime_explainer.Z
        Zl = lime_explainer.Zlr

        store_fcpn(i2e, results_filename, neigh_filename, dataset, black_box, fidelity, compact, compact_var,
                   lcompact, lcompact_var, plausibility, run_time, Z, Zl, 'rnd')


if __name__ == "__main__":
    main()
