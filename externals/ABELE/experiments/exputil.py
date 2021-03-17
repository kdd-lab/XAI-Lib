import json
import gzip
import pickle
import numpy as np

from skimage.color import gray2rgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from numpy import linalg as LA
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation
from tensorflow.keras.datasets import mnist, cifar10, fashion_mnist

from ..autoencoders.adversarial import AdversarialAutoencoderMnist, AdversarialAutoencoderCifar10
from ..autoencoders.variational import VariationalAutoencoderMnist, VariationalAutoencoderCifar10

import warnings
warnings.filterwarnings('ignore')


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def get_dataset(dataset):
    if dataset == 'mnist':
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        X_train = np.stack([gray2rgb(x) for x in X_train.reshape((-1, 28, 28))], 0)
        X_test = np.stack([gray2rgb(x) for x in X_test.reshape((-1, 28, 28))], 0)
        use_rgb = False

    elif dataset == 'cifar10':
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        Y_test = Y_test.ravel()
        use_rgb = True

    elif dataset == 'cifar10bw':
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        X_train = np.stack([rgb2gray(x) for x in X_train], 0)
        X_test = np.stack([rgb2gray(x) for x in X_test], 0)
        X_train = np.stack([gray2rgb(x) for x in X_train.reshape((-1, 32, 32))], 0)
        X_test = np.stack([gray2rgb(x) for x in X_test.reshape((-1, 32, 32))], 0)
        Y_test = Y_test.ravel()
        use_rgb = False

    elif dataset == 'fashion':
        (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
        X_train = np.stack([gray2rgb(x) for x in X_train.reshape((-1, 28, 28))], 0)
        X_test = np.stack([gray2rgb(x) for x in X_test.reshape((-1, 28, 28))], 0)
        use_rgb = False

    else:
        print('unknown dataset %s' % dataset)
        return -1

    return X_train, Y_train, X_test, Y_test, use_rgb


def get_black_box(black_box, black_box_filename, use_rgb, return_model=False):
    if black_box in ['RF', 'AB']:

        bb = pickle.load(open(black_box_filename + '.pickle', 'rb'))

        def bb_predict(X):
            X = np.array([rgb2gray(x) for x in X]) if not use_rgb else X
            X = X.astype('float32') / 255.
            X = np.array([x.ravel() for x in X])
            Y = bb.predict(X)
            return Y

        def bb_predict_proba(X):
            X = np.array([rgb2gray(x) for x in X]) if not use_rgb else X
            X = X.astype('float32') / 255.
            X = np.array([x.ravel() for x in X])
            Y = bb.predict_proba(X)
            return Y

        def transform(X):
            X = np.array([rgb2gray(x) for x in X]) if not use_rgb else X
            X = X.astype('float32') / 255.
            X = np.array([x.ravel() for x in X])
            return X

    elif black_box == 'DNN':

        model_filename = black_box_filename + '.json'
        weights_filename = black_box_filename + '_weights.hdf5'
        bb = model_from_json(open(model_filename, 'r').read())
        bb.load_weights(weights_filename)

        def bb_predict(X):
            X = X.astype('float32') / 255.
            Y = bb.predict(X)
            return np.argmax(Y, axis=1)

        def bb_predict_proba(X):
            X = X.astype('float32') / 255.
            Y = bb.predict(X)
            return Y

        def transform(X):
            X = X.astype('float32') / 255.
            return X

    else:
        print('unknown black box %s' % black_box)
        return -1

    if return_model:
        return bb, transform

    return bb_predict, bb_predict_proba


def build_dnn(input_shape, num_classes):
    model = Sequential()
    # add Convolutional layers
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # Densely connected layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    # output layer
    model.add(Dense(num_classes, activation='softmax'))
    # model.add(Dense(num_classes))
    # model.add(Activation('softmax'))
    # compile with adam optimizer & categorical_crossentropy loss function
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def train_black_box(X_train, Y_train, dataset, black_box, black_box_filename, use_rgb, random_state):
    if black_box in ['RF', 'AB']:

        if black_box == 'RF':
            model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_leaf=10,
                                           max_features='auto', random_state=random_state, n_jobs=-1)
        else:
            model = AdaBoostClassifier(n_estimators=100, random_state=random_state)

        def bb_fit(X, Y):
            X = np.array([rgb2gray(x) for x in X]) if not use_rgb else X
            X = X.astype('float32') / 255.
            X = np.array([x.ravel() for x in X])
            model.fit(X, Y)
            return model

        bb = bb_fit(X_train, Y_train)

        pickle_file = open(black_box_filename + '.pickle', 'wb')
        pickle.dump(bb, pickle_file)
        pickle_file.close()

    elif black_box == 'DNN':

        shape = X_train[0].shape
        num_classes = len(np.unique(Y_train))
        model = build_dnn(shape, num_classes)

        def bb_fit(X, Y):
            X = X.astype('float32') / 255.
            Y = to_categorical(Y, num_classes)
            model.fit(X, Y, epochs=100, batch_size=128, validation_split=0.3, verbose=True)
            return model

        bb = bb_fit(X_train, Y_train)

        model_filename = black_box_filename + '.json'
        weights_filename = black_box_filename + '_weights.hdf5'
        options = {'file_arch': model_filename, 'file_weight': weights_filename}
        json_string = bb.to_json()
        open(options['file_arch'], 'w').write(json_string)
        bb.save_weights(options['file_weight'])
    else:
        print('unknown black box %s' % dataset)
        return -1


def get_autoencoder(X, ae_name, dataset, path_aemodels):
    shape = X[0].shape
    input_dim = np.prod(X[0].shape)
    verbose = True
    store_intermediate = True

    if dataset == 'mnist':
        latent_dim = 4
    elif dataset == 'fashion':
        latent_dim = 8
    elif dataset == 'cifar10':
        latent_dim = 16
    elif dataset == 'cifar10bw':
        latent_dim = 16
    else:
        return

    name = '%s_%s_%d' % (ae_name, dataset, latent_dim)

    if ae_name == 'aae':
        if dataset in ['mnist', 'fashion', 'cifar10bw']:
            ae = AdversarialAutoencoderMnist(shape=shape, input_dim=input_dim, latent_dim=latent_dim,
                                             hidden_dim=1024, verbose=verbose, store_intermediate=store_intermediate,
                                             path=path_aemodels, name=name)
        elif dataset in ['cifar10']:
            ae = AdversarialAutoencoderCifar10(shape=shape, input_dim=input_dim, latent_dim=latent_dim,
                                               hidden_dim=128, verbose=verbose, store_intermediate=store_intermediate,
                                               path=path_aemodels, name=name)
        else:
            return -1

    elif ae_name == 'vae':
        if dataset in ['mnist', 'fashion']:
            ae = VariationalAutoencoderMnist(shape=shape, input_dim=input_dim, latent_dim=latent_dim, verbose=verbose,
                                             store_intermediate=store_intermediate, path=path_aemodels, name=name)

        elif dataset in ['cifar10']:
            ae = VariationalAutoencoderCifar10(shape=shape, input_dim=input_dim, latent_dim=latent_dim,
                                               hidden_dim=128, verbose=verbose, store_intermediate=store_intermediate,
                                               path=path_aemodels, name=name)

        else:
            return -1

    else:
        print('unknown autoencoder %s' % ae_name)
        return -1

    return ae


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def calculate_plausibilities(rdist, dist, nbr_real_instances):
    pllist = list()

    dist_thr = np.min(rdist)
    plausibility = len(dist[np.where(dist < dist_thr)]) / nbr_real_instances
    pllist.append(plausibility)
    for i in range(10, 100, 10):
        dist_thr = np.percentile(rdist, i)
        plausibility = len(dist[np.where(dist < dist_thr)]) / nbr_real_instances
        pllist.append(plausibility)
    dist_thr = np.max(rdist)
    plausibility = len(dist[np.where(dist < dist_thr)]) / nbr_real_instances
    pllist.append(plausibility)

    return pllist

# import matplotlib.pyplot as plt


def apply_relevancy(jrow_rel, img, bb_predict, bbo, relevancy, color):
    thr = np.min(relevancy)
    mask = np.where(relevancy < thr)
    fimg = img.copy()
    fimg[mask] = color
    # plt.imshow(fimg, cmap='gray')
    # plt.show()
    bbom = bb_predict(np.array([fimg]))[0]
    changed = 1 if bbo != bbom else 0
    jrow_rel['color%s_c0' % color] = changed

    for i in range(5, 100, 5):
        thr = np.percentile(relevancy, i)
        mask = np.where(relevancy < thr)
        fimg = img.copy()
        fimg[mask] = color
        # if i % 10 == 0:
        #     plt.imshow(fimg, cmap='gray')
        #     plt.show()
        bbom = bb_predict(np.array([fimg]))[0]
        changed = 1 if bbo != bbom else 0
        jrow_rel['color%s_c%d' % (color, i)] = changed

    thr = np.max(relevancy)
    mask = np.where(relevancy <= thr)
    fimg = img.copy()
    fimg[mask] = color
    # plt.imshow(fimg, cmap='gray')
    # plt.show()
    bbom = bb_predict(np.array([fimg]))[0]
    changed = 1 if bbo != bbom else 0
    jrow_rel['color%s_c100' % color] = changed

    return jrow_rel


def calculate_lipschitz_factor(x, x1):
    scaler = MinMaxScaler()
    x_s = scaler.fit_transform(x.ravel().reshape(-1, 1))
    x1_s = scaler.fit_transform(x1.ravel().reshape(-1, 1))
    norm = LA.norm(x_s - x1_s)
    return norm


def generate_random_noise(img, bb_predict, bbo, nbr_samples=20, max_attempts=100000):
    X = list()
    attempts = 0
    while len(X) < nbr_samples and attempts < max_attempts:
        attempts += 1
        imgr = img.copy()
        # imgr = imgr.astype('float32')
        # imgr += np.random.normal(0, 1, imgr.shape)
        # imgr = (imgr - np.min(imgr)) / (np.max(imgr) - np.min(imgr)) * 255
        pad = 11
        noise = np.random.randint(pad, size=imgr.shape[:2])
        imgr[np.where(noise == 0)] = 0
        imgr[np.where(noise == (pad - 1))] = 255
        bbor = bb_predict(np.array([imgr]))[0]
        if bbo == bbor:
            X.append(imgr.astype(np.int))

    X = np.array(X)
    return X


def store_fcpn(i2e, results_filename, neigh_filename, dataset, black_box, fidelity, compact, compact_var,
               lcompact, lcompact_var, plausibility, run_time, Z, Zl, neigh_type):

    jrow_fcp = {'i2e': i2e, 'dataset': dataset, 'black_box': black_box, 'fidelity': fidelity,
                'compactness': compact, 'compactness_var': compact_var,
                'lcompactness': lcompact, 'lcompactness_var': lcompact_var, 'time': run_time}

    for i, p in enumerate(plausibility):
        jrow_fcp['p%d' % i] = p

    jrow_fcp['neigh_type'] = neigh_type

    results = open(results_filename, 'a')
    results.write('%s\n' % json.dumps(jrow_fcp))
    results.close()

    # jrow_neigh = {'i2e': i2e, 'Z': Z, 'Zl': Zl}
    # json_str = ('%s\n' % json.dumps(jrow_neigh, cls=NumpyEncoder)).encode('utf-8')
    # with gzip.GzipFile(neigh_filename, 'a') as fout:
    #     fout.write(json_str)
