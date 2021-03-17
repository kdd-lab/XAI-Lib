import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


from experiments.exputil import *


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


# def rgb2gray(rgb):
#     return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

from skimage.color import rgb2grey, rgb2gray


def main():

    dataset = 'mnist'
    black_box = 'RF'

    ae_name = 'aae'
    latent_dim = 4
    verbose = True
    store_intermediate = True

    # path = '/Users/riccardo/Documents/PhD/ExplainImageClassifier/code/'
    path = '/home/riccardo/Documenti/PhD/ExplainingImageClassifiers/code/'
    path_models = path + 'models/'
    path_aemodels = path + 'aemodels/%s/%s/' % (dataset, ae_name)
    name = '%s_%s_%d' % (ae_name, dataset, latent_dim)

    black_box_filename = path_models + '%s_%s' % (dataset, black_box)

    _, _, X_test, _, use_rgb = get_dataset(dataset)

    bb_predict, bb_predict_proba = get_black_box(black_box, black_box_filename, use_rgb)

    ae = get_autoencoder(X_test, ae_name, dataset, latent_dim, verbose, store_intermediate, path_aemodels, name)
    ae.load_model()

    lX = ae.encode(X_test)
    X_pred = ae.decode(lX)

    print(X_pred[0].shape)
    print(np.min(X_pred[0]), np.max(X_pred[0]), np.mean(X_pred[0]))

    g = rgb2gray(X_pred[0])
    print(g.shape)
    print(np.min(g), np.max(g), np.mean(g))

    g = rgb2grey(X_pred[0])
    print(g.shape)
    print(np.min(g), np.max(g), np.mean(g))

    # print(np.unique(X_pred[0][:, :, 0]))
    # print(np.unique(X_pred[0][:, :, 1]))
    # print(np.unique(X_pred[0][:, :, 2]))
    # print(X_pred[0][:,:,1])
    # print(np.unique(rgb2gray(X_pred[0])))
    # Y_test = bb_predict(X_test)
    # Y_pred = bb_predict(X_pred)
    #
    # # valutazioen qualitativa
    # r, c = 5, 5
    #
    # cnt = 0
    # fig, axs = plt.subplots(r, c)
    # for i in range(r):
    #     for j in range(c):
    #         if not use_rgb:
    #             axs[i, j].imshow(X_test[cnt].reshape(X_test[cnt].shape), cmap='gray')
    #         else:
    #             axs[i, j].imshow(X_test[cnt].reshape(X_test[cnt].shape))
    #         axs[i, j].axis('off')
    #         cnt += 1
    # plt.show()
    # plt.close()
    #
    # cnt = 0
    # fig, axs = plt.subplots(r, c)
    # for i in range(r):
    #     for j in range(c):
    #         if not use_rgb:
    #             axs[i, j].imshow(X_pred[cnt].reshape(X_pred[cnt].shape), cmap='gray')
    #         else:
    #             axs[i, j].imshow(X_pred[cnt].reshape(X_pred[cnt].shape))
    #         axs[i, j].axis('off')
    #         cnt += 1
    # plt.show()
    # plt.close()
    #
    #
    # valutazione quantitativa
    # img_error = np.sum(np.abs(X_test - X_pred))/255/np.prod(X_pred.shape)
    # label_error = 1-accuracy_score(Y_test, Y_pred)
    # print(img_error, label_error)


if __name__ == '__main__':
    main()
