import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from skimage.color import rgb2gray

from keras.datasets import mnist

from experiments.exputil import *
from autoencoders.adversarial import AdversarialAutoencoderMnist


def generate_valid(autoencoder, valid_thr=0.5, lK=None):
    while True:
        if lK is None:
            lx = np.random.normal(size=(1, autoencoder.latent_dim))
        else:
            lx = np.random.normal(loc=np.mean(lK, axis=0), scale=np.std(lK, axis=0), size=(1, autoencoder.latent_dim))
        if autoencoder.discriminator is not None and valid_thr > 0.0:
            discriminator_out = autoencoder.discriminator.predict(lx)[0][0]
            if discriminator_out > valid_thr:
                return lx
        else:
            return lx


def generate_valid_samples(autoencoder, nbr_samples=1000, valid_thr=0.5):
    X = list()
    while len(X) < nbr_samples:
        X.append(generate_valid(autoencoder, valid_thr))
    return np.array(X).reshape(nbr_samples, autoencoder.latent_dim)


import warnings
warnings.filterwarnings('ignore')


def main():

    random_state = 0
    dataset = 'mnist'
    black_box = 'RF'

    ae_name = 'aae'
    latent_dim = 4
    verbose = True
    store_intermediate = True
    # epochs = 10000
    # batch_size = 256
    # sample_interval = 200

    # path = '/Users/riccardo/Documents/PhD/ExplainImageClassifier/code/'
    path = '/home/riccardo/Documenti/PhD/ExplainingImageClassifiers/code/'
    path_models = path + 'models/'
    path_aemodels = path + 'aemodels/%s/%s/' % (dataset, ae_name)
    name = '%s_%s_%d' % (ae_name, dataset, latent_dim)

    black_box_filename = path_models + '%s_%s' % (dataset, black_box)

    _, _, X_test, Y_test, use_rgb = get_dataset(dataset)

    bb_predict, bb_predict_proba = get_black_box(black_box, black_box_filename, use_rgb)

    ae = get_autoencoder(X_test, ae_name, dataset, latent_dim, verbose, store_intermediate, path_aemodels, name)
    ae.load_model()

    # visualizzazione spazio latente da dati reali - train
    X = X_test
    lX = ae.encode(X)
    Y = Y_test

    # visualizzazioen spazio latente da dati reali - test
    # X = np.reshape(X_train, [-1, input_dim])
    # lX = ae.encode(np.reshape(X_train, [-1, input_dim]))
    # Y = Y_train
    # print(np.mean(lX0, axis=0), np.std(lX0, axis=0))

    # visualizzazioen spazio latente da generati
    # lX = generate_valid_samples(ae, nbr_samples=1000, valid_thr=0.9, lK=None)
    # print(np.mean(lX, axis=0), np.std(lX, axis=0))

    # lX = generate_valid_samples(ae, nbr_samples=1000, valid_thr=0.9)
    # print(np.mean(lX, axis=0), np.std(lX, axis=0))

    X = ae.decode(lX)
    # Y = bb_predict(X)

    plt_idx = 1
    plt.figure(figsize=(12, 8))
    for dim0 in range(ae.latent_dim):
        for dim1 in range(dim0+1, ae.latent_dim):
            plt.subplot(2, 3, plt_idx)
            plt.scatter(lX[:, dim0], lX[:, dim1], c=Y, alpha=0.7)
            for v in np.unique(Y):
                lXv = lX[np.where(Y == v)]
                c = np.mean(lXv, axis=0)
                plt.plot(c[dim0], c[dim1], ms=10, marker='.', color='k')
                plt.text(c[dim0], c[dim1], v, fontsize=20)
                plt.xlabel('dim %d' % dim0)
                plt.ylabel('dim %d' % dim1)
            plt_idx += 1
    plt.show()

    # nx = ny = 20
    # x_values = np.linspace(-3, 3, nx)
    # y_values = np.linspace(-3, 3, ny)
    #
    # canvas = np.empty((28 * ny, 28 * nx))
    # for i, yi in enumerate(x_values):
    #     for j, xi in enumerate(y_values):
    #         z_mu = np.array([xi, yi] * 2)
    #         print(z_mu.shape)
    #         x_mean = ae.decode(z_mu.reshape(1, -1))
    #         canvas[(nx - i - 1) * 28:(nx - i) * 28, j * 28:(j + 1) * 28] = x_mean[0].reshape(28, 28)
    #
    # plt.figure(figsize=(8, 10))
    # Xi, Yi = np.meshgrid(x_values, y_values)
    # plt.imshow(canvas, origin="upper", cmap="gray")
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    main()
