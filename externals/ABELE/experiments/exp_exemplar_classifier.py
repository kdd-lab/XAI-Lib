import sys

import copy
import gzip
import json
import datetime
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from ilore.ilorem import ILOREM
from ilore.util import neuclidean

from experiments.mmd import calculate_kernel_individual
from experiments.mmd import greedy_select_protos
from experiments.mmd import select_criticism_regularized

from experiments.exputil import get_dataset
from experiments.exputil import get_black_box
from experiments.exputil import get_autoencoder


def prepare_data_for_knn(exemplars, cexemplars, Ye, Yc):
    X_train_knn = list()
    X_train_knn.extend(exemplars)
    X_train_knn.extend(cexemplars)
    X_train_knn = np.array(X_train_knn)
    Y_train_knn = list()
    Y_train_knn.extend(Ye)
    Y_train_knn.extend(Yc)
    Y_train_knn = np.array(Y_train_knn)
    return X_train_knn, Y_train_knn


def prepare_test_for_knn(bbo, X_test_comp, Y_pred_comp, bb_predict, instance_to_test=200):
    # X_idx = np.where(Y_pred_comp == bbo[0])[0]
    # scaler = MinMaxScaler()
    # x0 = scaler.fit_transform(img.ravel().reshape(-1, 1))
    # Xj = scaler.fit_transform([x.ravel() for x in X_test_comp[X_idx]])
    # dist = cdist(x0.reshape(1, -1), Xj)[0]
    # eps = np.percentile(dist, 5)
    # X_idx_eps_pos = X_idx[np.where(dist <= eps)]
    #
    # X_idx = np.where(Y_pred_comp != bbo[0])[0]
    # scaler = MinMaxScaler()
    # x0 = scaler.fit_transform(img.ravel().reshape(-1, 1))
    # Xj = scaler.fit_transform([x.ravel() for x in X_test_comp[X_idx]])
    # dist = cdist(x0.reshape(1, -1), Xj)[0]
    # eps = np.percentile(dist, 5)
    # X_idx_eps_neg = X_idx[np.where(dist <= eps)]
    #
    # X_idx_eps = np.concatenate([X_idx_eps_pos, X_idx_eps_neg])
    # X_test_knn = X_test_comp[X_idx_eps]
    # Y_test_knn = bb_predict(X_test_knn)

    X_idx_pos = np.where(Y_pred_comp == bbo[0])[0]
    X_idx_neg = np.where(Y_pred_comp == bbo[0])[0]
    itt = instance_to_test // 2
    X_idx = np.concatenate([np.random.choice(X_idx_pos, itt), np.random.choice(X_idx_neg, itt)])
    X_test_knn = X_test_comp[X_idx]
    Y_test_knn = bb_predict(X_test_knn)
    return X_test_knn, Y_test_knn


def evaluate_with_knn(X_train, Y_train, X_test, Y_test, k=1):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit([x.ravel() for x in X_train], Y_train)
    Y_pred = clf.predict([x.ravel() for x in X_test])
    knn_eval = accuracy_score(Y_test, Y_pred)
    return knn_eval


def main():

    dataset = sys.argv[1]

    # dataset = 'mnist'

    if len(sys.argv) > 2:
        start_from = int(sys.argv[2])
    else:
        start_from = 0


    black_box = 'DNN'
    neigh_type = 'hrgp'
    max_nbr_exemplars = 128

    random_state = 0
    ae_name = 'aae'
    gamma = None

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
    path_results = path + 'results/expcl/'
    path_aemodels = path + 'aemodels/%s/%s/' % (dataset, ae_name)

    black_box_filename = path_models + '%s_%s' % (dataset, black_box)
    results_filename = path_results + 'expcl_%s_%s_%s.json' % (dataset, black_box, neigh_type)

    _, _, X_test, Y_test, use_rgb = get_dataset(dataset)
    bb, transform = get_black_box(black_box, black_box_filename, use_rgb, return_model=True)
    bb_predict, bb_predict_proba = get_black_box(black_box, black_box_filename, use_rgb)

    Y_pred = bb_predict(X_test)

    X_test_comp = X_test[nbr_experiments:]
    Y_pred_comp = Y_pred[nbr_experiments:]
    X = np.array([x.ravel() for x in transform(X_test_comp)])

    ae = get_autoencoder(X_test, ae_name, dataset, path_aemodels)
    ae.load_model()

    class_name = 'class'
    class_values = ['%s' % i for i in range(len(np.unique(Y_test)))]

    explainer = ILOREM(bb_predict, class_name, class_values, neigh_type=neigh_type, use_prob=True, size=1000, ocr=0.1,
                       kernel_width=None, kernel=None, autoencoder=ae, use_rgb=use_rgb, valid_thr=0.5,
                       filter_crules=True, random_state=random_state, verbose=False, alpha1=0.5, alpha2=0.5,
                       metric=neuclidean, ngen=10, mutpb=0.2, cxpb=0.5, tournsize=3, halloffame_ratio=0.1,
                       bb_predict_proba=bb_predict_proba)

    errors = open(path_results + 'errors_expcl_%s_%s.csv' % (dataset, black_box), 'w')

    for i2e in range(nbr_experiments):

        if i2e < start_from:
            continue

        print(datetime.datetime.now(), '[%s/%s] %s %s' % (i2e, nbr_experiments, dataset, black_box))

        try:

            img = X_test[i2e]
            bbo = bb_predict(np.array([img]))
            jrow_o = {'i2e': i2e, 'dataset': dataset, 'black_box': black_box}
            jrow_list = list()

            X_test_knn, Y_test_knn = prepare_test_for_knn(bbo, X_test_comp, Y_pred_comp, bb_predict, 200)

            # Alore
            print(datetime.datetime.now(), 'alore')
            exp = explainer.explain_instance(img, num_samples=1000, use_weights=True, metric=neuclidean)

            # MMD
            # kernel = rbf_kernel(X, gamma=gamma)
            print(datetime.datetime.now(), 'mmd')
            kernel = calculate_kernel_individual(X, Y_pred_comp, gamma)

            # K-Medoids
            print(datetime.datetime.now(), 'k-medoids')
            X_idx = np.where(Y_pred_comp == bbo)[0]
            Z = X[X_idx]
            scaler = MinMaxScaler()
            Z = scaler.fit_transform(Z)
            zimg = scaler.transform(transform(np.array([img])).ravel().reshape(1, -1))
            dist = cdist(zimg.reshape(1, -1), Z)
            idx_p = np.argsort(dist)
            idx_p = np.array([X_idx[i] for i in idx_p])
            idx_p = idx_p[np.where(idx_p != i2e)]

            X_idx = np.where(Y_pred_comp != bbo)[0]
            Z = X[X_idx]
            scaler = MinMaxScaler()
            Z = scaler.fit_transform(Z)
            zimg = scaler.transform(transform(np.array([img])).ravel().reshape(1, -1))
            dist = cdist(zimg.reshape(1, -1), Z)
            idx_n = np.argsort(dist)
            idx_n = np.array([X_idx[i] for i in idx_n])
            idx_n = idx_n[np.where(idx_n != i2e)]

            # Alore
            print(datetime.datetime.now(), 'alore - generate exemplars')
            alore_exemplars = exp.get_prototypes_respecting_rule(num_prototypes=max_nbr_exemplars)
            alore_cexemplars = exp.get_counterfactual_prototypes(eps=0.01)
            if len(alore_cexemplars) < max_nbr_exemplars:
                cexemplars2 = exp.get_prototypes_not_respecting_rule(num_prototypes=max_nbr_exemplars - len(alore_cexemplars))
                alore_cexemplars.extend(cexemplars2)

            # MMD
            print(datetime.datetime.now(), 'mmd - select prototypes')
            proto_idx = greedy_select_protos(kernel, np.array(range(np.shape(kernel)[0])),
                                             nbr_prototypes=max_nbr_exemplars)
            crit_idx = select_criticism_regularized(kernel, proto_idx, nbr_criticisms=max_nbr_exemplars,
                                                    is_K_sparse=False, reg='logdet')

            proto_idx_sel = [i for i in proto_idx if Y_pred_comp[i] == bbo[0]][:max_nbr_exemplars]
            crit_idx_sel = [i for i in crit_idx if Y_pred_comp[i] != bbo[0]][:max_nbr_exemplars]
            mmd_exemplars = X_test_comp[proto_idx_sel]
            mmd_cexemplars = X_test_comp[crit_idx_sel]

            # K-Medoids
            print(datetime.datetime.now(), 'k-medoids - select prototypes')
            kmedoids_exemplars = X_test_comp[idx_p[:max_nbr_exemplars]]
            kmedoids_cexemplars = X_test_comp[idx_n[:max_nbr_exemplars]]

            for nbr_exemplars in [1, 2, 4, 8, 16, 32, 64, 128]:

                jrow_e = copy.deepcopy(jrow_o)
                jrow_e['nbr_exemplars'] = nbr_exemplars

                X_train_knn, Y_train_knn = prepare_data_for_knn(alore_exemplars[:nbr_exemplars],
                                                                alore_cexemplars[:nbr_exemplars],
                                                                bb_predict(np.array(alore_exemplars[:nbr_exemplars])),
                                                                bb_predict(np.array(alore_cexemplars[:nbr_exemplars])))

                for k in range(1, min(nbr_exemplars + 1, 11)):
                    jrow = copy.deepcopy(jrow_e)
                    acc = evaluate_with_knn(X_train_knn, Y_train_knn, X_test_knn, Y_test_knn, k)
                    jrow['method'] = 'alore'
                    jrow['k'] = k
                    jrow['accuracy'] = acc
                    jrow_list.append(jrow)
                    print(datetime.datetime.now(),
                          '[%s/%s] %s %s %s - alore: %.3f' % (i2e, nbr_experiments, dataset, black_box, k, acc))

                X_train_knn, Y_train_knn = prepare_data_for_knn(mmd_exemplars[:nbr_exemplars],
                                                                mmd_cexemplars[:nbr_exemplars],
                                                                bb_predict(np.array(mmd_exemplars[:nbr_exemplars])),
                                                                bb_predict(np.array(mmd_cexemplars[:nbr_exemplars])))

                for k in range(1, min(nbr_exemplars + 1, 11)):
                    jrow = copy.deepcopy(jrow_e)
                    acc = evaluate_with_knn(X_train_knn, Y_train_knn, X_test_knn, Y_test_knn, k)
                    jrow['method'] = 'mmd'
                    jrow['k'] = k
                    jrow['accuracy'] = acc
                    jrow_list.append(jrow)
                    print(datetime.datetime.now(),
                          '[%s/%s] %s %s %s - mmd: %.3f' % (i2e, nbr_experiments, dataset, black_box, k, acc))

                X_train_knn, Y_train_knn = prepare_data_for_knn(kmedoids_exemplars[:nbr_exemplars],
                                                                kmedoids_cexemplars[:nbr_exemplars],
                                                                bb_predict(np.array(kmedoids_exemplars[:nbr_exemplars])),
                                                                bb_predict(np.array(kmedoids_cexemplars[:nbr_exemplars])))

                for k in range(1, min(nbr_exemplars + 1, 11)):
                    jrow = copy.deepcopy(jrow_e)
                    acc = evaluate_with_knn(X_train_knn, Y_train_knn, X_test_knn, Y_test_knn, k)
                    jrow['method'] = 'k-medoids'
                    jrow['k'] = k
                    jrow['accuracy'] = acc
                    jrow_list.append(jrow)
                    print(datetime.datetime.now(),
                          '[%s/%s] %s %s %s - k-medoids: %.3f' % (i2e, nbr_experiments, dataset, black_box, k, acc))

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
