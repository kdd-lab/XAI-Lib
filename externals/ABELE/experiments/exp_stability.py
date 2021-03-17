import sys

import copy
import json
import datetime
import numpy as np

from collections import defaultdict

from ilore.ilorem import ILOREM
from ilore.util import neuclidean

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

from keras.models import Model
from keras import backend as K
from keras.utils import to_categorical
from deepexplain.tensorflow import DeepExplain

from experiments.exputil import get_dataset
from experiments.exputil import get_black_box
from experiments.exputil import get_autoencoder
from experiments.exputil import calculate_lipschitz_factor
from experiments.exputil import generate_random_noise


def main():

    dataset = sys.argv[1]

    black_box = 'DNN'
    neigh_type = 'hrgp'

    if len(sys.argv) > 2:
        start_from = int(sys.argv[2])
    else:
        start_from = 0

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
    path_results = path + 'results/stability/'
    path_aemodels = path + 'aemodels/%s/%s/' % (dataset, ae_name)

    black_box_filename = path_models + '%s_%s' % (dataset, black_box)
    results_filename = path_results + 'sta_%s_%s_%s.json' % (dataset, black_box, neigh_type)

    _, _, X_test, Y_test, use_rgb = get_dataset(dataset)
    bb, transform = get_black_box(black_box, black_box_filename, use_rgb, return_model=True)
    bb_predict, bb_predict_proba = get_black_box(black_box, black_box_filename, use_rgb)

    Y_pred = bb_predict(X_test)
    Y_pred_proba = bb_predict_proba(X_test)

    ae = get_autoencoder(X_test, ae_name, dataset, path_aemodels)
    ae.load_model()

    class_name = 'class'
    class_values = ['%s' % i for i in range(len(np.unique(Y_test)))]

    explainer = ILOREM(bb_predict, class_name, class_values, neigh_type=neigh_type, use_prob=True, size=1000, ocr=0.1,
                       kernel_width=None, kernel=None, autoencoder=ae, use_rgb=use_rgb, valid_thr=0.5,
                       filter_crules=True, random_state=random_state, verbose=False, alpha1=0.5, alpha2=0.5,
                       metric=neuclidean, ngen=10, mutpb=0.2, cxpb=0.5, tournsize=3, halloffame_ratio=0.1,
                       bb_predict_proba=bb_predict_proba)

    lime_explainer = lime_image.LimeImageExplainer()
    segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

    input_tensor = bb.layers[0].input
    last_layer = -2 if dataset == 'mnist' else -1
    bb_model = Model(inputs=input_tensor, outputs=bb.layers[last_layer].output)
    target_tensor = bb_model(input_tensor)
    de_list = ['grad*input', 'saliency', 'intgrad', 'elrp', 'occlusion']

    errors = open(path_results + 'errors_stability_%s_%s.csv' % (dataset, black_box), 'w')

    with DeepExplain(session=K.get_session()) as de:

        for i2e in range(nbr_experiments):

            if i2e < start_from:
                continue

            try:

                print(datetime.datetime.now(), '[%s/%s] %s %s - checking stability' % (
                    i2e, nbr_experiments, dataset, black_box))

                expl_list = list()
                jrow_list = list()

                jrow_coh_o = {'i2e': i2e, 'dataset': dataset, 'black_box': black_box}

                # Crate random noise
                img = X_test[i2e]
                bbo = bb_predict(np.array([img]))
                bbop = Y_pred_proba[i2e]
                X_random_noise = generate_random_noise(img, bb_predict, bbo[0], nbr_samples=20)
                # Y_pred_random_noise = bb_predict(X_random_noise)
                Y_pred_proba_random_noise = bb_predict_proba(X_random_noise)

                # plt.subplot(1, 3, 1)
                # plt.imshow(X_random_noise[0], cmap='gray')
                # plt.subplot(1, 3, 2)
                # plt.imshow(X_random_noise[1], cmap='gray')
                # plt.subplot(1, 3, 3)
                # plt.imshow(X_random_noise[2], cmap='gray')
                # plt.show()

                # Alore
                print(datetime.datetime.now(), 'calculating alore')
                exp = explainer.explain_instance(img, num_samples=1000, use_weights=True, metric=neuclidean)
                _, diff = exp.get_image_rule(features=None, samples=100)
                expl_list.append(diff)

                # Lime
                print(datetime.datetime.now(), 'calculating lime')
                exp = lime_explainer.explain_instance(img, bb_predict_proba, top_labels=1, hide_color=0,
                                                      num_samples=1000, segmentation_fn=segmenter)
                _, mask = exp.get_image_and_mask(bbo[0], positive_only=False, num_features=5, hide_rest=False,
                                                 min_weight=0.01)
                expl_list.append(mask)

                # Deep Explain
                xs = transform(np.array([img]))
                ys = to_categorical(bbo, num_classes)

                for det in de_list:
                    print(datetime.datetime.now(), 'calculating %s' % det)
                    if det == 'shapley_sampling':
                        maps = de.explain(det, target_tensor, input_tensor, xs, ys=ys, samples=10)[0]
                    else:
                        maps = de.explain(det, target_tensor, input_tensor, xs, ys=ys)[0]
                    maps = np.mean(maps, axis=2)
                    expl_list.append(maps)

                lipschitz_list = defaultdict(list)
                lipschitz_list_bb = defaultdict(list)

                print(datetime.datetime.now(), 'calculating lipschitz')
                for i2e1 in range(len(X_random_noise)):
                    img1 = X_random_noise[i2e1]
                    bbo1 = bb_predict(np.array([img1]))
                    bbop1 = Y_pred_proba_random_noise[i2e1]
                    norm_bb = calculate_lipschitz_factor(bbop, bbop1)
                    norm_x = calculate_lipschitz_factor(img, img1)

                    # Alore
                    exp1 = explainer.explain_instance(img1, num_samples=1000, use_weights=True, metric=neuclidean)
                    _, diff1 = exp1.get_image_rule(features=None, samples=100)

                    norm_exp = calculate_lipschitz_factor(expl_list[0], diff1)
                    lipschitz_list['alore'].append(norm_exp / norm_x)
                    lipschitz_list_bb['alore'].append(norm_exp / norm_bb)
                    print(datetime.datetime.now(), '\talore', norm_exp / norm_x)

                    # Lime
                    exp1 = lime_explainer.explain_instance(img1, bb_predict_proba, top_labels=1, hide_color=0,
                                                           num_samples=1000, segmentation_fn=segmenter)
                    _, mask1 = exp1.get_image_and_mask(bbo[0], positive_only=False, num_features=5, hide_rest=False,
                                                       min_weight=0.01)
                    norm_exp = calculate_lipschitz_factor(expl_list[1], mask1)
                    lipschitz_list['lime'].append(norm_exp / norm_x)
                    lipschitz_list_bb['lime'].append(norm_exp / norm_bb)
                    print(datetime.datetime.now(), '\tlime', norm_exp / norm_x)

                    # DeepExplain
                    xs1 = transform(np.array([img1]))
                    ys1 = to_categorical(bbo1, num_classes)

                    for i, det in enumerate(de_list):
                        if det == 'shapley_sampling':
                            maps1 = de.explain(det, target_tensor, input_tensor, xs1, ys=ys1, samples=10)[0]
                        else:
                            maps1 = de.explain(det, target_tensor, input_tensor, xs1, ys=ys1)[0]
                        maps1 = np.mean(maps1, axis=2)
                        norm_exp = calculate_lipschitz_factor(expl_list[i+2], maps1)
                        lipschitz_list[det].append(norm_exp / norm_x)
                        lipschitz_list_bb[det].append(norm_exp / norm_bb)
                        print(datetime.datetime.now(), '\t%s' % det, norm_exp / norm_x)

                for k in lipschitz_list:
                    jrow_coh = copy.deepcopy(jrow_coh_o)
                    jrow_coh['method'] = k
                    jrow_coh['mean'] = float(np.nanmean(lipschitz_list[k]))
                    jrow_coh['std'] = float(np.nanstd(lipschitz_list[k]))
                    jrow_coh['max'] = float(np.nanmax(lipschitz_list[k]))
                    jrow_coh['mean_bb'] = float(np.nanmean(lipschitz_list_bb[k]))
                    jrow_coh['std_bb'] = float(np.nanstd(lipschitz_list_bb[k]))
                    jrow_coh['max_bb'] = float(np.nanmax(lipschitz_list_bb[k]))
                    jrow_list.append(jrow_coh)
                    print(datetime.datetime.now(),
                          '[%s/%s] %s %s %s - mean: %.3f, max: %.3f' % (i2e, nbr_experiments, dataset, black_box, k, 
                                                                        jrow_coh['mean'], jrow_coh['max']))

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
