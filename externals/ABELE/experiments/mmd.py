import sys

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler

from experiments.exputil import get_dataset
from experiments.exputil import get_black_box


def greedy_select_protos(K, candidate_indices, nbr_prototypes, is_K_sparse=False):

    if len(candidate_indices) != np.shape(K)[0]:
        K = K[:, candidate_indices][candidate_indices, :]

    n = len(candidate_indices)

    # colsum = np.array(K.sum(0)).ravel() # same as rowsum
    if is_K_sparse:
        colsum = 2*np.array(K.sum(0)).ravel() / n
    else:
        colsum = 2*np.sum(K, axis=0) / n

    selected = np.array([], dtype=int)
    # value = np.array([])
    for i in range(nbr_prototypes):
        # maxx = -sys.float_info.max
        # argmax = -1
        candidates = np.setdiff1d(range(n), selected)

        s1array = colsum[candidates]
        if len(selected) > 0:
            temp = K[selected, :][:, candidates]
            if is_K_sparse:
                # s2array = temp.sum(0) *2
                s2array = temp.sum(0) * 2 + K.diagonal()[candidates]

            else:
                s2array = np.sum(temp, axis=0) *2 + np.diagonal(K)[candidates]

            s2array = s2array/(len(selected) + 1)

            s1array = s1array - s2array

        else:
            if is_K_sparse:
                s1array = s1array - (np.abs(K.diagonal()[candidates]))
            else:
                s1array = s1array - (np.abs(np.diagonal(K)[candidates]))

        argmax = candidates[np.argmax(s1array)]
        # print "max %f" %np.max(s1array)

        selected = np.append(selected, argmax)
        # value = np.append(value,maxx)
        # KK = K[selected, :][:, selected]
        # if is_K_sparse:
        #     KK = KK.todense()

        # inverse_of_prev_selected = np.linalg.inv(KK)  # shortcut

    return candidate_indices[selected]


def select_criticism_regularized(K, selectedprotos, nbr_criticisms, reg='logdet', is_K_sparse=True):

    n = np.shape(K)[0]
    if reg in ['None','logdet','iterative']:
        pass
    else:
        print("wrong regularizer :" + reg)
        exit(1)
    # options = dict()

    selected = np.array([], dtype=int)
    candidates2 = np.setdiff1d(range(n), selectedprotos)
    inverse_of_prev_selected = None  # should be a matrix

    if is_K_sparse:
        colsum = np.array(K.sum(0)).ravel()/n
    else:
        colsum = np.sum(K, axis=0)/n

    for i in range(nbr_criticisms):
        # maxx = -sys.float_info.max
        # argmax = -1
        candidates = np.setdiff1d(candidates2, selected)

        s1array = colsum[candidates]

        temp = K[selectedprotos, :][:, candidates]
        if is_K_sparse:
            s2array = temp.sum(0)
        else:
            s2array = np.sum(temp, axis=0)

        s2array = s2array / (len(selectedprotos))

        s1array = np.abs(s1array - s2array)
        if reg == 'logdet':
            if inverse_of_prev_selected is not None:  # first call has been made already
                temp = K[selected, :][:, candidates]
                if is_K_sparse:
                    temp2 = temp.transpose().dot(inverse_of_prev_selected)
                    regularizer = temp.transpose().multiply(temp2)
                    regcolsum = regularizer.sum(1).ravel()  # np.sum(regularizer, axis=0)
                    regularizer = np.abs(K.diagonal()[candidates] - regcolsum)

                else:
                    # hadamard product
                    temp2 = np.array(np.dot(inverse_of_prev_selected, temp))
                    regularizer = temp2 * temp
                    regcolsum = np.sum(regularizer, axis=0)
                    regularizer = np.log(np.abs(np.diagonal(K)[candidates] - regcolsum))
                s1array = s1array + regularizer
            else:
                if is_K_sparse:
                    s1array = s1array - np.log(np.abs(K.diagonal()[candidates]))
                else:
                    s1array = s1array - np.log(np.abs(np.diagonal(K)[candidates]))
        argmax = candidates[np.argmax(s1array)]
        # maxx = np.max(s1array)

        selected = np.append(selected, argmax)
        if reg == 'logdet':
            KK = K[selected, :][:, selected]
            if is_K_sparse:
                KK = KK.todense()

            inverse_of_prev_selected = np.linalg.inv(KK) # shortcut
        if reg == 'iterative':
            selectedprotos = np.append(selectedprotos, argmax)

    return selected


def calculate_kernel_individual(X, Y, gamma=None):
    kernel = np.zeros((np.shape(X)[0], np.shape(X)[0]))
    sortind = np.argsort(Y)
    X = X[sortind, :]
    Y = Y[sortind]

    for i in np.unique(Y):
        ind = np.where(Y == i)[0]
        startind = np.min(ind)
        endind = np.max(ind)+1
        kernel[startind:endind, startind:endind] = rbf_kernel(X[startind:endind, :], gamma=gamma)

    return kernel


def main():

    dataset = 'mnist'
    black_box = 'RF'
    gamma = None
    nbr_prototypes = 10
    nbr_criticisms = 10

    path = './'
    path_models = path + 'models/'

    black_box_filename = path_models + '%s_%s' % (dataset, black_box)

    _, _, X_test, Y_test, use_rgb = get_dataset(dataset)
    bb, transform = get_black_box(black_box, black_box_filename, use_rgb, return_model=True)
    bb_predict, _ = get_black_box(black_box, black_box_filename, use_rgb)

    i2e = 0
    img = X_test[i2e]
    Y_pred = bb_predict(X_test)
    bbo = Y_pred[i2e]

    # for label in np.unique(Y_pred):
    X_idx = np.where(Y_pred == bbo)[0]
    Z = transform(X_test[X_idx])
    scaler = MinMaxScaler()
    Z = scaler.fit_transform(Z)
    zimg = scaler.transform(transform(np.array([img])))
    dist = cdist(zimg.reshape(1, -1), Z)
    idx = np.argsort(dist)
    idx = np.array([X_idx[i] for i in idx])
    idx = idx[np.where(idx != i2e)]


    print(Y_pred[i2e])
    plt.imshow(X_test[i2e])
    plt.show()

    plt.imshow(X_test[idx[0]])
    plt.show()

    plt.imshow(X_test[idx[-1]])
    plt.show()



    # kernel = rbf_kernel(transform(X_test), gamma=gamma)
    # kernel = calculate_kernel_individual(transform(X_test), Y_pred, gamma)
    #
    # proto_idx = greedy_select_protos(kernel, np.array(range(np.shape(kernel)[0])), nbr_prototypes)
    #
    # crit_idx = select_criticism_regularized(kernel, proto_idx, nbr_criticisms, is_K_sparse=False, reg='logdet')
    #
    # i = 2
    # print(Y_pred[i])
    # plt.imshow(X_test[proto_idx[i]])
    # plt.show()
    #
    # plt.imshow(X_test[crit_idx[i]])
    # plt.show()


if __name__ == "__main__":
    main()
