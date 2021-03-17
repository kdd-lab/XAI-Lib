import sys

import datetime
import numpy as np

from scipy.spatial.distance import cdist

from ilore.ilorem import ILOREM
from ilore.util import neuclidean

from experiments.exputil import get_dataset
from experiments.exputil import get_black_box
from experiments.exputil import get_autoencoder
from experiments.exputil import store_fcpn
from experiments.exputil import calculate_plausibilities


def main():

    dataset = sys.argv[1]
    black_box = sys.argv[2]
    neigh_type = sys.argv[3]
    if len(sys.argv) > 4:
        start_from = int(sys.argv[4])
    else:
        start_from = 0

    # dataset = 'mnist'
    # black_box = 'DNN'
    # neigh_type = 'hrgp'

    random_state = 0
    ae_name = 'aae'

    nbr_experiments = 200

    if dataset not in ['mnist', 'cifar10', 'fashion']:
        print('unknown dataset %s' % dataset)
        return -1

    if black_box not in ['RF', 'AB', 'DNN']:
        print('unknown black box %s' % black_box)
        return -1

    if neigh_type not in ['rnd', 'gntp', 'hrgp']:
        print('unknown neigh type %s' % neigh_type)
        return -1

    path = './'
    path_models = path + 'models/'
    path_results = path + 'results/fcp/'
    path_aemodels = path + 'aemodels/%s/%s/' % (dataset, ae_name)
    path_neigh = './neigh/'

    black_box_filename = path_models + '%s_%s' % (dataset, black_box)
    results_filename = path_results + 'alore_p_%s_%s_%s.json' % (dataset, black_box, neigh_type)
    neigh_filename = path_neigh + 'alore_%s_%s_%s.json.gz' % (dataset, black_box, neigh_type)

    _, _, X_test, Y_test, use_rgb = get_dataset(dataset)
    bb, transform = get_black_box(black_box, black_box_filename, use_rgb, return_model=True)
    bb_predict, bb_predict_proba = get_black_box(black_box, black_box_filename, use_rgb)
    ae = get_autoencoder(X_test, ae_name, dataset, path_aemodels)
    ae.load_model()

    class_name = 'class'
    class_values = ['%s' % i for i in range(len(np.unique(Y_test)))]

    explainer = ILOREM(bb_predict, class_name, class_values, neigh_type=neigh_type, use_prob=True, size=1000, ocr=0.1,
                       kernel_width=None, kernel=None, autoencoder=ae, use_rgb=use_rgb, valid_thr=0.5,
                       filter_crules=True, random_state=random_state, verbose=False, alpha1=0.5, alpha2=0.5,
                       metric=neuclidean, ngen=10, mutpb=0.2, cxpb=0.5, tournsize=3, halloffame_ratio=0.1,
                       bb_predict_proba=bb_predict_proba)

    errors = open(path_results + 'errors_alore_%s_%s_%s.csv' % (dataset, black_box, neigh_type), 'w')

    for i2e in range(nbr_experiments):
        if i2e < start_from:
            continue

        img = X_test[i2e]

        start_time = datetime.datetime.now()
        try:
            exp = explainer.explain_instance(img, num_samples=1000, use_weights=True, metric=neuclidean)
        except Exception:
            print('error instance to explain: %d' % i2e)
            errors.write('%d\n' % i2e)
            errors.flush()
            continue

        run_time = (datetime.datetime.now() - start_time).total_seconds()

        fidelity = exp.fidelity

        img_cdist = transform(np.array([img]))
        Zl = exp.Z
        Z = exp.autoencoder.decode(Zl)
        Z_cdist = transform(Z)

        if black_box == 'DNN':
            img_cdist = np.array([x.ravel() for x in img_cdist])
            Z_cdist = np.array([x.ravel() for x in Z_cdist])

        rdist = cdist(img_cdist, Z_cdist, metric='euclidean')
        compact, compact_var = float(np.mean(rdist)), float(np.std(rdist))

        sdist = cdist(np.array([exp.Z[0]]), exp.Z, metric=neuclidean)
        lcompact, lcompact_var = float(np.mean(sdist)), float(np.std(sdist))

        X_test_cdist = transform(X_test)
        if black_box == 'DNN':
            X_test_cdist = np.array([x.ravel() for x in X_test_cdist])

        dist = cdist(img_cdist, X_test_cdist, metric='euclidean')
        nbr_real_instances = len(X_test)
        plausibility = calculate_plausibilities(rdist, dist, nbr_real_instances)

        print(datetime.datetime.now(), '[%s/%s] %s %s %s - f: %.2f, c: %.2f, lc: %.2f, p: %.2f' % (
            i2e, nbr_experiments, dataset, black_box, neigh_type, fidelity, compact, lcompact, plausibility[-2]))

        store_fcpn(i2e, results_filename, neigh_filename, dataset, black_box, fidelity, compact, compact_var,
                   lcompact, lcompact_var, plausibility, run_time, Z, Zl, neigh_type)

    errors.close()


if __name__ == "__main__":
    main()
