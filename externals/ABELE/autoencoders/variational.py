import numpy as np
from datetime import datetime

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape, LeakyReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import plot_model

from tensorflow.keras.datasets import mnist

from abc import abstractmethod

from ..autoencoders.autoencoder import Autoencoder


def sampling(args):
    mu, log_var = args
    epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
    return mu + K.exp(log_var / 2) * epsilon


class VariationalAutoencoder(Autoencoder):

    def __init__(self, shape, input_dim, latent_dim=4, hidden_dim=1024, alpha=0.2, verbose=False,
                 store_intermediate=False, save_graph=False, path='./', name='vae'):
        super(VariationalAutoencoder, self).__init__(shape, input_dim, latent_dim, hidden_dim, alpha, verbose,
                                                     store_intermediate, save_graph, path, name)

    @abstractmethod
    def img_normalize(self, X):
        return

    @abstractmethod
    def img_denormalize(self, X):
        return

    @abstractmethod
    def init(self):
        return

    def fit(self, X_train, epochs=30000, batch_size=128, sample_interval=200):

        self.init()
        X_train = self.img_normalize(X_train)

        # validation_split = 0.3, train_on_batch_flag = True
        # if not train_on_batch_flag:
        #     self.autoencoder.fit(X_train, X_train, shuffle=True, epochs=epochs, batch_size=batch_size,
        #                          validation_split=validation_split)
        # else:
        past = datetime.now()

        for epoch in range(epochs):

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            Xs = X_train[idx]

            # idx_test = np.random.randint(0, X_test.shape[0], batch_size)
            # Xs_test = X_test[idx_test]

            vae_loss = self.autoencoder.train_on_batch(Xs, Xs)

            now = datetime.now()
            # Plot the progress
            if self.verbose and epoch % sample_interval == 0:
                print("Epoch %d/%d, %.2f [A loss: %f]" % (epoch, epochs, (now - past).total_seconds(), vae_loss))
            past = now

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0 and self.store_intermediate:
                self.sample_images(epoch)
                self.save_model()

        self.save_model()


class VariationalAutoencoderMnist(VariationalAutoencoder):

    def __init__(self, shape, input_dim, latent_dim=4, hidden_dim=1024, alpha=0.2, verbose=False,
                 store_intermediate=False, save_graph=False, path='./', name='vae'):
        super(VariationalAutoencoderMnist, self).__init__(shape, input_dim, latent_dim, hidden_dim, alpha, verbose,
                                                          store_intermediate, save_graph, path, name)

    def img_normalize(self, X):
        return X.astype(np.float32) / 255.0

    def img_denormalize(self, X):
        return (X * 255).astype(np.int)

    def init(self):

        # Build Encoder
        # x = Input(shape=(self.input_dim, ))
        x = Input(shape=self.shape)
        eh = Flatten()(x)
        eh = Dense(self.hidden_dim)(eh)
        eh = LeakyReLU(alpha=self.alpha)(eh)
        eh = Dense(self.hidden_dim)(eh)
        eh = LeakyReLU(alpha=self.alpha)(eh)
        mu = Dense(self.latent_dim)(eh)
        log_var = Dense(self.latent_dim)(eh)
        latent_repr = Lambda(sampling)([mu, log_var])
        self.encoder = Model(x, [mu, log_var, latent_repr])

        # Build Decoder
        di = Input(shape=(self.latent_dim, ))
        dh = Dense(self.hidden_dim)(di)
        dh = LeakyReLU(alpha=self.alpha)(dh)
        dh = Dense(self.hidden_dim)(dh)
        dh = LeakyReLU(alpha=self.alpha)(dh)
        do = Dense(self.input_dim, activation='sigmoid')(dh)
        do = Reshape(self.shape)(do)
        self.decoder = Model(di, do)

        # Build Vae
        outputs = self.decoder(self.encoder(x)[2])
        self.autoencoder = Model(x, outputs)

        def vae_loss(x, tx):
            x = K.flatten(x)
            tx = K.flatten(tx)
            xent_loss = binary_crossentropy(x, tx) * self.input_dim
            kl_loss = - 0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)

        self.autoencoder.compile(optimizer='adam', loss=vae_loss)
        self.autoencoder.summary()

        if self.save_graph:
            plot_model(self.encoder, to_file='%s%s_encoder.png' % (self.path, self.name))
            plot_model(self.decoder, to_file='%s%s_decoder.png' % (self.path, self.name))
            plot_model(self.autoencoder, to_file='%s%s_vae.png' % (self.path, self.name))


class VariationalAutoencoderCifar10(VariationalAutoencoder):

    def __init__(self, shape, input_dim, latent_dim=4, hidden_dim=1024, alpha=0.2, verbose=False,
                 store_intermediate=False, save_graph=False, path='./', name='vae'):
        super(VariationalAutoencoderCifar10, self).__init__(shape, input_dim, latent_dim, hidden_dim, alpha, verbose,
                                                            store_intermediate, save_graph, path, name)

    def img_normalize(self, X):
        return X.astype(np.float32) / 255.0

    def img_denormalize(self, X):
        return (X * 255).astype(np.int)

    def init(self):

        # Build Encoder
        x = Input(shape=self.shape)

        # eh = Conv2D(64, (3, 3), padding='same')(x)
        # eh = BatchNormalization()(eh)
        # eh = Activation('relu')(eh)
        # eh = MaxPooling2D((2, 2), padding='same')(eh)
        #
        # eh = Conv2D(32, (3, 3), padding='same')(eh)
        # eh = BatchNormalization()(eh)
        # eh = Activation('relu')(eh)
        # eh = MaxPooling2D((2, 2), padding='same')(eh)
        #
        # eh = Conv2D(16, (3, 3), padding='same')(eh)
        # eh = BatchNormalization()(eh)
        # eh = Activation('relu')(eh)
        # eh = MaxPooling2D((2, 2), padding='same')(eh)

        eh = Conv2D(3, kernel_size=(2, 2), padding='same', activation='relu')(x)
        eh = Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu', strides=(2, 2))(eh)
        eh = Conv2D(32, kernel_size=3, padding='same', activation='relu', strides=1)(eh)
        eh = Conv2D(32, kernel_size=3, padding='same', activation='relu', strides=1)(eh)
        eh = Flatten()(eh)
        eh = Dense(self.hidden_dim, activation='relu')(eh)

        mu = Dense(self.latent_dim)(eh)
        log_var = Dense(self.latent_dim)(eh)
        lx = Lambda(sampling, output_shape=(self.latent_dim,))([mu, log_var])

        self.encoder = Model(x, lx)
        if self.verbose:
            self.encoder.summary()

        # Build Decoder

        lx = Input(shape=(self.latent_dim,))

        dh = Dense(self.hidden_dim, activation='relu')(lx)
        # dh = Dense(self.hidden_dim * 2, activation='relu')(dh)
        # dh = Reshape((4, 4, 16))(dh)

        # dh = Conv2D(16, (3, 3), padding='same')(dh)
        # dh = BatchNormalization()(dh)
        # dh = Activation('relu')(dh)
        # dh = UpSampling2D((2, 2))(dh)
        #
        # dh = Conv2D(32, (3, 3), padding='same')(dh)
        # dh = BatchNormalization()(dh)
        # dh = Activation('relu')(dh)
        # dh = UpSampling2D((2, 2))(dh)
        #
        # dh = Conv2D(64, (3, 3), padding='same')(dh)
        # dh = BatchNormalization()(dh)
        # dh = Activation('relu')(dh)
        # dh = UpSampling2D((2, 2))(dh)
        #
        # dh = Conv2D(3, (3, 3), padding='same')(dh)
        # dh = BatchNormalization()(dh)
        # dh = Activation('sigmoid')(dh)

        dh = Dense(32 * 16 * 16, activation='relu')(dh)
        dh = Reshape((16, 16, 32))(dh)
        dh = Conv2DTranspose(32, kernel_size=3, padding='same', strides=1, activation='relu')(dh)
        dh = Conv2DTranspose(32, kernel_size=3, padding='same', strides=1, activation='relu')(dh)
        dh = Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')(dh)
        dh = Conv2D(3, kernel_size=2, padding='valid', activation='sigmoid')(dh)

        self.decoder = Model(lx, dh)
        if self.verbose:
            self.decoder.summary()

        # Build Vae
        outputs = self.decoder(self.encoder(x))
        self.autoencoder = Model(x, outputs)

        def vae_loss(x, tx):
            x = K.flatten(x)
            tx = K.flatten(tx)
            xent_loss = binary_crossentropy(x, tx) * self.input_dim
            kl_loss = - 0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)

        self.autoencoder.compile(optimizer='rmsprop', loss=vae_loss)  #'adam'
        self.autoencoder.summary()

        if self.save_graph:
            plot_model(self.encoder, to_file='%s%s_encoder.png' % (self.path, self.name))
            plot_model(self.decoder, to_file='%s%s_decoder.png' % (self.path, self.name))
            plot_model(self.autoencoder, to_file='%s%s_vae.png' % (self.path, self.name))


def main():

    # np.random.seed(0)
    # # Load the dataset
    # (X_train, _), (_, _) = mnist.load_data()
    # shape = X_train[0].shape
    #
    # # Rescale -1 to 1
    # X_train = (X_train.astype(np.float32)) / 255
    # input_dim = np.prod(X_train[0].shape)
    # X_train = np.reshape(X_train, [-1, input_dim])

    # test_ratio = 0.3
    # test_size = int(np.round(len(X_train) * test_ratio))
    # test_idx = np.random.choice(len(X_train), test_size, replace=False)
    # train_idx = np.array(list(set(np.arange(len(X_train))) - set(test_idx)))
    # X_train, X_test = X_train[train_idx], X_train[test_idx]

    # Load the dataset
    (_, _), (X_test, Y_test) = mnist.load_data()

    shape = X_test[0].shape

    # Rescale -1 to 1
    # X = img_normalize(X_test)
    input_dim = np.prod(X_test[0].shape)
    X = np.reshape(X_test, [-1, input_dim])

    latent_dim = 4
    verbose = True
    store_intermediate = True

    path = './mnist/vae/'
    name = 'mnist_vae_a%d' % latent_dim

    epochs = 10000
    batch_size = 128
    sample_interval = 200

    vae = VariationalAutoencoder(shape=shape, input_dim=input_dim, latent_dim=latent_dim, verbose=verbose,
                                 store_intermediate=store_intermediate, path=path, name=name)

    vae.fit(X, epochs=epochs, batch_size=batch_size, sample_interval=sample_interval)  #, train_on_batch_flag=False)
    # aae.save_model()

    # vae.load_model()
    vae.sample_images(epochs)


if __name__ == '__main__':
    main()
