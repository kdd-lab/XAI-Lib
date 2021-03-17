import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json

from abc import abstractmethod


def save(model, model_name, path):
    model_filename = '%s%s.json' % (path, model_name)
    weights_filename = '%s%s_weights.hdf5' % (path, model_name)
    options = {'file_arch': model_filename, 'file_weight': weights_filename}
    json_string = model.to_json()
    open(options['file_arch'], 'w').write(json_string)
    model.save_weights(options['file_weight'])


def load(model_name, path):
    model_filename = "%s%s.json" % (path, model_name)
    weights_filename = "%s%s_weights.hdf5" % (path, model_name)
    model = model_from_json(open(model_filename, 'r').read())
    model.load_weights(weights_filename)
    return model


class Autoencoder(object):

    def __init__(self, shape, input_dim, latent_dim=4, hidden_dim=1024, alpha=0.2, verbose=False,
                 store_intermediate=False, save_graph=False, path='./', name='aae'):
        self.shape = shape
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.verbose = verbose
        self.store_intermediate = store_intermediate
        self.save_graph = save_graph
        self.path = path
        self.name = name

        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.discriminator = None
        self.generator = None

    @abstractmethod
    def init(self):
        return

    @abstractmethod
    def fit(self, X, epochs=30000, batch_size=128, sample_interval=100):
        return

    @abstractmethod
    def img_normalize(self, X):
        return

    @abstractmethod
    def img_denormalize(self, X):
        return

    def encode(self, X):
        return self.encoder.predict(self.img_normalize(X))

    def decode(self, lX):
        return self.img_denormalize(self.decoder.predict(lX))

    def discriminate(self, lX):
        if self.discriminator is None:
            return None
        return self.discriminator.predict(lX)

    def generate(self, X):
        if self.discriminator is None:
            return None
        return self.img_denormalize(self.discriminator.predict(self.img_normalize(X)))

    def sample_images(self, epoch):
        r, c = 5, 5

        z = np.random.normal(size=(r*c, self.latent_dim))
        Z = self.decoder.predict(z)
        Z = self.img_denormalize(Z)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                if self.shape[2] == 1:
                    axs[i, j].imshow(Z[cnt].reshape(self.shape), cmap='gray')
                else:
                    axs[i, j].imshow(Z[cnt].reshape(self.shape))
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig('%s%s_%d.png' % (self.path, self.name, epoch))
        plt.close()

    def save_model(self):
        save(self.encoder, '%s_encoder' % self.name, self.path)
        save(self.decoder, '%s_decoder' % self.name, self.path)
        save(self.autoencoder, '%s_autoencoder' % self.name, self.path)

        if self.discriminator is not None:
            save(self.discriminator, '%s_discriminator' % self.name, self.path)
        if self.generator is not None:
            save(self.generator, '%s_generator' % self.name, self.path)

    def load_model(self):
        self.encoder = load('%s_encoder' % self.name, self.path)
        self.decoder = load('%s_decoder' % self.name, self.path)
        self.autoencoder = load('%s_autoencoder' % self.name, self.path)
        if 'aae' in self.name or 'aag' in self.name:
            self.discriminator = load('%s_discriminator' % self.name, self.path)
        if 'aag' in self.name:
            self.generator = load('%s_generator' % self.name, self.path)

