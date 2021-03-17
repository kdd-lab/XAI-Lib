import sys

import copy
import json
import datetime
import numpy as np

from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

from ilore.ilorem import ILOREM
from ilore.util import neuclidean

from experiments.exputil import get_dataset
from experiments.exputil import get_black_box
from experiments.exputil import get_autoencoder
from experiments.exputil import calculate_plausibilities

from experiments.exp_relevancy_mapsgen import apply_relevancy
from experiments.exp_coherence import calculate_lipschitz_factor
from experiments.exp_stability import generate_random_noise
from experiments.exp_exemplar_classifier import prepare_test_for_knn
from experiments.exp_exemplar_classifier import prepare_data_for_knn
from experiments.exp_exemplar_classifier import evaluate_with_knn


def get_compactness(img, exp, black_box, transform):
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

    return compact, compact_var, lcompact, lcompact_var, img_cdist, rdist


def get_plausibility(img_cdist, black_box, transform, X_test, rdist):
    X_test_cdist = transform(X_test)
    if black_box == 'DNN':
        X_test_cdist = np.array([x.ravel() for x in X_test_cdist])

    dist = cdist(img_cdist, X_test_cdist, metric='euclidean')
    nbr_real_instances = len(X_test)
    plausibility = calculate_plausibilities(rdist, dist, nbr_real_instances)
    return plausibility


def finding_lipswhitz_neigh(img, i2e, bb_predict, X_test_comp, Y_pred_comp, Y_pred_proba):
    bbo = bb_predict(np.array([img]))
    X_idx = np.where(Y_pred_comp == bbo[0])[0]

    scaler = MinMaxScaler()
    x0 = scaler.fit_transform(img.ravel().reshape(-1, 1))
    Xj = scaler.fit_transform([x.ravel() for x in X_test_comp[X_idx]])
    dist = cdist(x0.reshape(1, -1), Xj)[0]
    eps = np.percentile(dist, 5)
    X_idx_eps = X_idx[np.where(dist <= eps)]
    return X_idx_eps


def get_lipswhitz_coherence(img, i2e, bb_predict, X_test_comp, Y_pred_comp, Y_pred_proba, Y_pred_proba_comp, diff,
                            explainer):
    # Finding Lipswhitz neighborhood
    X_idx_eps = finding_lipswhitz_neigh(img, i2e, bb_predict, X_test_comp, Y_pred_comp, Y_pred_proba)
    bbop = Y_pred_proba[i2e]

    lipschitz_list = list()
    lipschitz_list_bb = list()
    for i2e1 in X_idx_eps[:20]:
        img1 = X_test_comp[i2e1]
        bbop1 = Y_pred_proba_comp[i2e1]
        norm_bb = calculate_lipschitz_factor(bbop, bbop1)
        norm_x = calculate_lipschitz_factor(img, img1)

        exp1 = explainer.explain_instance(img1, num_samples=1000, use_weights=True, metric=neuclidean)
        _, diff1 = exp1.get_image_rule(features=None, samples=100)

        norm_exp = calculate_lipschitz_factor(diff, diff1)
        lipschitz_list.append(norm_exp / norm_x)
        lipschitz_list_bb.append(norm_exp / norm_bb)

    return lipschitz_list, lipschitz_list_bb


def get_lipswhitz_stability(img, i2e, bb_predict, X_test_comp, Y_pred_proba, Y_pred_proba_comp, diff, explainer):
    # Finding random noise
    bbo = bb_predict(np.array([img]))
    X_random_noise = generate_random_noise(img, bb_predict, bbo[0], nbr_samples=20)
    bbop = Y_pred_proba[i2e]

    lipschitz_list = list()
    lipschitz_list_bb = list()
    for i2e1 in range(len(X_random_noise)):
        img1 = X_test_comp[i2e1]
        bbop1 = Y_pred_proba_comp[i2e1]
        norm_bb = calculate_lipschitz_factor(bbop, bbop1)
        norm_x = calculate_lipschitz_factor(img, img1)

        exp1 = explainer.explain_instance(img1, num_samples=1000, use_weights=True, metric=neuclidean)
        _, diff1 = exp1.get_image_rule(features=None, samples=100)

        norm_exp = calculate_lipschitz_factor(diff, diff1)
        lipschitz_list.append(norm_exp / norm_x)
        lipschitz_list_bb.append(norm_exp / norm_bb)

    return lipschitz_list, lipschitz_list_bb


def main():

    dataset = sys.argv[1]
    black_box = sys.argv[2]
    neigh_type = sys.argv[3]

    # dataset = 'mnist'
    # black_box = 'DNN'
    # neigh_type = 'hrgp'
    max_nbr_exemplars = 128

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
    path_results = path + 'results/validity/'
    path_aemodels = path + 'aemodels/%s/%s/' % (dataset, ae_name)

    black_box_filename = path_models + '%s_%s' % (dataset, black_box)
    results_filename = path_results + 'validthr_%s_%s_%s.json' % (dataset, black_box, neigh_type)

    _, _, X_test, Y_test, use_rgb = get_dataset(dataset)
    bb, transform = get_black_box(black_box, black_box_filename, use_rgb, return_model=True)
    bb_predict, bb_predict_proba = get_black_box(black_box, black_box_filename, use_rgb)
    ae = get_autoencoder(X_test, ae_name, dataset, path_aemodels)
    ae.load_model()

    Y_pred = bb_predict(X_test)
    Y_pred_proba = bb_predict_proba(X_test)

    X_test_comp = X_test[nbr_experiments:]
    Y_pred_comp = Y_pred[nbr_experiments:]
    Y_pred_proba_comp = Y_pred_proba[nbr_experiments:]

    class_name = 'class'
    class_values = ['%s' % i for i in range(len(np.unique(Y_test)))]

    errors = open(path_results + 'errors_validity_%s_%s_%s.csv' % (dataset, black_box, neigh_type), 'w')

    for i2e in range(nbr_experiments):
        print(datetime.datetime.now(), '[%s/%s] %s %s' % (i2e, nbr_experiments, dataset, black_box))

        for valid_thr in np.arange(0.0, 1.0, 0.1):

            explainer = ILOREM(bb_predict, class_name, class_values, neigh_type=neigh_type, use_prob=True, size=1000,
                               ocr=0.1, kernel_width=None, kernel=None, autoencoder=ae, use_rgb=use_rgb,
                               valid_thr=valid_thr, filter_crules=True, random_state=random_state, verbose=False,
                               alpha1=0.5, alpha2=0.5, metric=neuclidean, ngen=10, mutpb=0.2, cxpb=0.5, tournsize=3,
                               halloffame_ratio=0.1, bb_predict_proba=bb_predict_proba)

            try:

                jrow_o = {'i2e': i2e, 'dataset': dataset, 'black_box': black_box, 'neigh_type': neigh_type,
                          'valid_thr': valid_thr}
                jrow_list = list()

                img = X_test[i2e]
                bbo = bb_predict(np.array([img]))

                start_time = datetime.datetime.now()

                print(datetime.datetime.now(), '\textract explanation', valid_thr)
                exp = explainer.explain_instance(img, num_samples=1000, use_weights=True, metric=neuclidean)

                run_time = (datetime.datetime.now() - start_time).total_seconds()
                jrow_o['time'] = run_time

                fidelity = exp.fidelity
                jrow_o['fidelity'] = fidelity

                print(datetime.datetime.now(), '\tcalculate compactness', valid_thr)
                res_compactness = get_compactness(img, exp, black_box, transform)
                compact, compact_var, lcompact, lcompact_var, img_cdist, rdist = res_compactness
                jrow_o['compactness'] = compact
                jrow_o['compactness_var'] = compact_var
                jrow_o['lcompactness'] = lcompact
                jrow_o['lcompactness_var'] = lcompact_var

                print(datetime.datetime.now(), '\tcalculate plausibility', valid_thr)
                plausibility = get_plausibility(img_cdist, black_box, transform, X_test, rdist)
                for i, p in enumerate(plausibility):
                    jrow_o['plausibility%d' % i] = p

                print(datetime.datetime.now(), '\tcalculate saliency map', valid_thr)
                _, diff = exp.get_image_rule(features=None, samples=100)

                print(datetime.datetime.now(), '\tcalculate relevancy', valid_thr)
                relevancy = 1.0 - np.abs(diff - 127.5) / 127.5
                for color in [0, 127, 255]:
                    jrow_o = apply_relevancy(jrow_o, img, bb_predict, bbo[0], relevancy, color)

                print(datetime.datetime.now(), '\tcalculate coherence', valid_thr)
                coherence_lipschitz, coherence_lipschitz_bb = get_lipswhitz_coherence(img, i2e, bb_predict, X_test_comp,
                                                                                      Y_pred_comp, Y_pred_proba,
                                                                                      Y_pred_proba_comp, diff,
                                                                                      explainer)
                jrow_o['coherence_mean'] = float(np.nanmean(coherence_lipschitz))
                jrow_o['coherence_std'] = float(np.nanstd(coherence_lipschitz))
                jrow_o['coherence_max'] = float(np.nanmax(coherence_lipschitz))
                jrow_o['coherence_mean_bb'] = float(np.nanmean(coherence_lipschitz_bb))
                jrow_o['coherence_std_bb'] = float(np.nanstd(coherence_lipschitz_bb))
                jrow_o['coherence_max_bb'] = float(np.nanmax(coherence_lipschitz_bb))

                print(datetime.datetime.now(), '\tcalculate stability', valid_thr)
                stability_lipschitz, stability_lipschitz_bb = get_lipswhitz_stability(img, i2e, bb_predict, X_test_comp,
                                                                                      Y_pred_proba, Y_pred_proba_comp,
                                                                                      diff,
                                                                                      explainer)

                jrow_o['stability_mean'] = float(np.nanmean(stability_lipschitz))
                jrow_o['stability_std'] = float(np.nanstd(stability_lipschitz))
                jrow_o['stability_max'] = float(np.nanmax(stability_lipschitz))
                jrow_o['stability_mean_bb'] = float(np.nanmean(stability_lipschitz_bb))
                jrow_o['stability_std_bb'] = float(np.nanstd(stability_lipschitz_bb))
                jrow_o['stability_max_bb'] = float(np.nanmax(stability_lipschitz_bb))

                print(datetime.datetime.now(), '\tcalculate knn', valid_thr)
                exemplars = exp.get_prototypes_respecting_rule(num_prototypes=max_nbr_exemplars)
                cexemplars = exp.get_counterfactual_prototypes(eps=0.01)
                if len(cexemplars) < max_nbr_exemplars:
                    cexemplars2 = exp.get_prototypes_not_respecting_rule(
                        num_prototypes=max_nbr_exemplars - len(cexemplars))
                    cexemplars.extend(cexemplars2)

                X_test_knn, Y_test_knn = prepare_test_for_knn(bbo, X_test_comp, Y_pred_comp, bb_predict, 200)
                for nbr_exemplars in [1, 2, 4, 8, 16, 32, 64, 128]:
                    jrow_e = copy.deepcopy(jrow_o)
                    jrow_e['nbr_exemplars'] = nbr_exemplars

                    X_train_knn, Y_train_knn = prepare_data_for_knn(exemplars[:nbr_exemplars],
                                                                    cexemplars[:nbr_exemplars],
                                                                    bb_predict(np.array(exemplars[:nbr_exemplars])),
                                                                    bb_predict(np.array(cexemplars[:nbr_exemplars])))
                    for k in range(1, min(nbr_exemplars + 1, 11)):
                        acc = evaluate_with_knn(X_train_knn, Y_train_knn, X_test_knn, Y_test_knn, k)
                        jrow = copy.deepcopy(jrow_e)
                        jrow['k'] = k
                        jrow['accuracy'] = acc
                        jrow_list.append(jrow)

                results = open(results_filename, 'a')
                for jrow in jrow_list:
                    results.write('%s\n' % json.dumps(jrow))
                results.close()

            except Exception:
                print('error instance to explain: %d' % i2e)
                errors.write('%d\n' % i2e)
                continue

    errors.close()


if __name__ == "__main__":
    main()
