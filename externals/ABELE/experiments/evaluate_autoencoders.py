import sys

import numpy as np
from experiments.exputil import get_dataset
from experiments.exputil import get_autoencoder

import warnings
warnings.filterwarnings('ignore')


def rmse(x, y):
    return np.sqrt(np.mean((x - y)**2))


def main():

    ae_name = 'aae'

    for dataset in ['mnist', 'fashion', 'cifar10']:

        path = './'
        path_aemodels = path + 'aemodels/%s/%s/' % (dataset, ae_name)

        X_train, _, X_test, _, use_rgb = get_dataset(dataset)
        ae = get_autoencoder(X_test, ae_name, dataset, path_aemodels)
        ae.load_model()

        X_train_ae = ae.decode(ae.encode(X_train))
        X_test_ae = ae.decode(ae.encode(X_test))

        print('dataset: ', dataset)
        print('train rmse:', rmse(X_train, X_train_ae))
        print('test rmse:', rmse(X_test, X_test_ae))
        print('')





if __name__ == "__main__":
    main()
