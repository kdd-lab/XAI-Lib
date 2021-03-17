import sys

import copy
import gzip
import json
import datetime
import numpy as np

import matplotlib.pyplot as plt

from ilore.ilorem import ILOREM
from ilore.util import neuclidean

from keras.models import Model
from keras import backend as K
from keras.utils import to_categorical
from deepexplain.tensorflow import DeepExplain

from experiments.exputil import get_dataset
from experiments.exputil import get_black_box
from experiments.exputil import get_autoencoder
from experiments.exputil import apply_relevancy
from experiments.exputil import NumpyEncoder


def main():

    dataset = sys.argv[1]

    black_box = 'RF'
    neigh_type = 'hrgp'

    random_state = 0
    ae_name = 'aae'
    num_classes = 10

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
    path_results = path + 'results/rel/'
    path_aemodels = path + 'aemodels/%s/%s/' % (dataset, ae_name)
    path_expl = './expl/'

    black_box_filename = path_models + '%s_%s' % (dataset, black_box)
    results_filename = path_results + 'rel_%s_%s_%s.json' % (dataset, black_box, neigh_type)
    expl_filename = path_expl + 'alore_%s_%s_%s.json.gz' % (dataset, black_box, neigh_type)

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

    errors = open(path_results + 'errors_relevancy_%s_%s.csv' % (dataset, black_box), 'w')

    for i2e in range(nbr_experiments):

        jrow_rel_o = {'i2e': i2e, 'dataset': dataset, 'black_box': black_box}

        img = X_test[i2e]
        bbo = bb_predict(np.array([img]))
        jrow_list = list()

        try:
            # Alore
            exp = explainer.explain_instance(img, num_samples=1000, use_weights=True, metric=neuclidean)
            _, diff = exp.get_image_rule(features=None, samples=100)
            relevancy = 1.0 - np.abs(diff - 127.5)/127.5

            for color in [0, 127, 255]:
                jrow_rel = copy.deepcopy(jrow_rel_o)
                jrow_rel['method'] = 'alore'
                jrow_rel['color'] = color
                jrow_rel = apply_relevancy(jrow_rel, img, bb_predict, bbo[0], relevancy, color)
                jrow_list.append(jrow_rel)
                print(datetime.datetime.now(),
                      '[%s/%s] %s %s %s %s - 25: %d, 50: %d, 75: %d' % (i2e, nbr_experiments, dataset, black_box,
                                                                        'alore', color, jrow_rel['color%s_c25' % color],
                                                                        jrow_rel['color%s_c50' % color], jrow_rel['color%s_c75' % color]))

            jrow_neigh = {'i2e': i2e, 'dataset': dataset, 'black_box': black_box, 'expl': diff,
                          'rule': str(exp.rule)}

            json_str = ('%s\n' % json.dumps(jrow_neigh, cls=NumpyEncoder)).encode('utf-8')
            with gzip.GzipFile(expl_filename, 'a') as fout:
                fout.write(json_str)

        except Exception:
            print('error instance to explain: %d' % i2e)
            errors.write('%d\n' % i2e)
            continue

        results = open(results_filename, 'a')
        for jrow in jrow_list:
            results.write('%s\n' % json.dumps(jrow))
        results.close()

    errors.close()


if __name__ == "__main__":
    main()
