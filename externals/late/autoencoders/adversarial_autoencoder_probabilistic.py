from tensorflow import keras
import tensorflow as tf
from late.autoencoders.adversarial_autoencoder import autoencoder_loss, generator_loss, discriminator_loss, \
    build_discriminator
from late.plots.plots import plot_latent_space_matrix, plot_latent_space
from late.autoencoders.variational_autoencoder import build_encoder, build_decoder, EncoderWrapper


class AAEP(keras.Model):
    def __init__(self, encoder, decoder, discriminator, latent_dim, **kwargs):
        super(AAEP, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.latent_dim = latent_dim

    def compile(
            self,
            disc_optimizer,
            gen_optimizer,
            aut_optimizer,
            disc_loss,
            gen_loss,
            aut_loss,
            disc_loss_weight=1,
            gen_loss_weight=1,
            aut_loss_weight=1,
            loss=None
    ):
        super(AAEP, self).compile()
        self.disc_optimizer = disc_optimizer
        self.gen_optimizer = gen_optimizer
        self.aut_optimizer = aut_optimizer
        self.disc_loss = disc_loss
        self.gen_loss = gen_loss
        self.aut_loss = aut_loss
        self.disc_loss_weight = disc_loss_weight
        self.gen_loss_weight = gen_loss_weight
        self.aut_loss_weight = aut_loss_weight
        self.disc_loss_tracker = keras.metrics.Mean(name="disc_loss")
        self.disc_acc_tracker = keras.metrics.BinaryAccuracy(name="disc_acc")
        self.gen_loss_tracker = keras.metrics.Mean(name="gen_loss")
        self.aut_loss_tracker = keras.metrics.Mean(name="aut_loss")

    def call(self, inputs, training=None, mask=None):
        return self.decoder(self.encoder(inputs)[2])

    @property
    def metrics(self):
        return [
            self.disc_loss_tracker,
            self.disc_acc_tracker,
            self.gen_loss_tracker,
            self.aut_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as aut_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            _, _, latent_fake = self.encoder(data, training=True)
            aut_loss = autoencoder_loss(data, self.decoder(latent_fake, training=True),
                                        self.aut_loss, self.aut_loss_weight)
            latent_real = tf.random.normal([tf.shape(data)[0], self.latent_dim])
            y_fake = self.discriminator(latent_fake, training=True)
            y_real = self.discriminator(latent_real, training=True)
            disc_loss = discriminator_loss(y_real, y_fake, self.disc_loss, self.disc_loss_weight)
            gen_loss = generator_loss(y_fake, self.gen_loss, self.gen_loss_weight)

        grads_aut = aut_tape.gradient(aut_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        self.aut_optimizer.apply_gradients(zip(grads_aut,
                                               self.encoder.trainable_variables + self.decoder.trainable_variables))
        grads_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(grads_disc, self.discriminator.trainable_variables))

        grads_gen = gen_tape.gradient(gen_loss, self.encoder.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(grads_gen, self.encoder.trainable_variables))

        self.disc_loss_tracker.update_state(disc_loss)
        self.disc_acc_tracker.update_state(
            tf.concat([tf.ones_like(y_real), tf.zeros_like(y_fake)], axis=0),
            tf.concat([y_real, y_fake], axis=0)
        )
        self.gen_loss_tracker.update_state(gen_loss)
        self.aut_loss_tracker.update_state(aut_loss)
        return {
            "disc_loss": self.disc_loss_tracker.result(),
            "disc_acc": self.disc_acc_tracker.result(),
            "gen_loss": self.gen_loss_tracker.result(),
            "aut_loss": self.aut_loss_tracker.result(),
        }

    def test_step(self, data):
        _, _, latent_fake = self.encoder(data, training=True)
        aut_loss = autoencoder_loss(data, self.decoder(latent_fake), self.aut_loss, self.aut_loss_weight)
        latent_real = tf.random.normal([tf.shape(data)[0], self.latent_dim])
        y_fake = self.discriminator(latent_fake)
        y_real = self.discriminator(latent_real)
        disc_loss = discriminator_loss(y_real, y_fake, self.disc_loss, self.disc_loss_weight)
        gen_loss = generator_loss(y_fake, self.gen_loss, self.gen_loss_weight)

        self.disc_loss_tracker.update_state(disc_loss)
        self.disc_acc_tracker.update_state(
            tf.concat([tf.ones_like(y_real), tf.zeros_like(y_fake)], axis=0),
            tf.concat([y_real, y_fake], axis=0)
        )
        self.gen_loss_tracker.update_state(gen_loss)
        self.aut_loss_tracker.update_state(aut_loss)

        return {
            "disc_loss": self.disc_loss_tracker.result(),
            "disc_acc": self.disc_acc_tracker.result(),
            "gen_loss": self.gen_loss_tracker.result(),
            "aut_loss": self.aut_loss_tracker.result(),
        }


def build_aaep(
        input_shape,
        latent_dim,
        autoencoder_kwargs,
        discriminator_kwargs,
        verbose=True,
):
    encoder = build_encoder(
        input_shape,
        latent_dim,
        **autoencoder_kwargs
    )
    encoder_w = EncoderWrapper(encoder)

    decoder = build_decoder(
        encoder,
        latent_dim,
        **autoencoder_kwargs
    )

    discriminator = build_discriminator(
        latent_dim,
        **discriminator_kwargs
    )

    autoencoder = AAEP(encoder, decoder, discriminator, latent_dim)
    autoencoder.compile(
        disc_optimizer=discriminator_kwargs.get("optimizer", keras.optimizers.Adam()),
        gen_optimizer=autoencoder_kwargs.get("optimizer", keras.optimizers.Adam()),
        aut_optimizer=autoencoder_kwargs.get("optimizer", keras.optimizers.Adam()),
        disc_loss=tf.keras.losses.MeanSquaredError(),
        gen_loss=tf.keras.losses.MeanSquaredError(),
        aut_loss=tf.keras.losses.MeanSquaredError(),
        disc_loss_weight=1,
        gen_loss_weight=1,
        aut_loss_weight=1,
        loss=None
    )
    autoencoder.build(input_shape[::-1])  # wants the n_timesteps on the -1 axis

    if verbose:
        encoder.summary()
        decoder.summary()
        discriminator.summary()
        autoencoder.summary()
        # encoder_discriminator.summary()

    return encoder_w, decoder, discriminator, autoencoder


if __name__ == "__main__":
    from late.datasets.datasets import build_cbf, load_ucr_dataset
    from late.blackboxes.loader import blackbox_loader
    from late.blackboxes.blackbox_wrapper import BlackboxWrapper
    from late.utils.utils import plot_reconstruction_vae
    import matplotlib.pyplot as plt

    (X_train, y_train, X_val, y_val,
     X_test, y_test, X_exp_train, y_exp_train,
     X_exp_val, y_exp_val, X_exp_test, y_exp_test) = build_cbf(n_samples=600)

    (X_train, y_train, X_val, y_val,
     X_test, y_test, X_exp_train, y_exp_train,
     X_exp_val, y_exp_val, X_exp_test, y_exp_test) = load_ucr_dataset("ArrowHead")

    input_shape = X_train.shape[1:]

    latent_dim = 2

    autoencoder_kwargs = {
        "filters": [4, 4, 4, 4],
        "kernel_size": [3, 3, 3, 3],
        "padding": ["same", "same", "same", "same"],
        "activation": ["relu", "relu", "relu", "relu"],
        "pooling": [1, 1, 1, 1],
        "n_layers": 4,
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.0001),
        "n_layers_residual": None,
        "batch_normalization": None,
    }

    autoencoder_kwargs = {
        "filters": [8, 8, 8, 8, 8, 8, 8, 8],
        "kernel_size": [3, 3, 3, 3, 3, 3, 3, 3],
        "padding": ["same", "same", "same", "same", "same", "same", "same", "same"],
        "activation": ["relu", "relu", "relu", "relu", "relu", "relu", "relu",
                       "relu"],
        "pooling": [1, 1, 1, 1, 1, 1, 1, 1],
        "n_layers": 8,
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.0001),
        "n_layers_residual": None,
        "batch_normalization": None,
    }

    discriminator_kwargs = {
        "n_layers": 2,
        "units": [100, 100],
        "activation": ["relu", "relu"],
        "optimizer": tf.keras.optimizers.Adam(learning_rate=0.0001),
        "loss": "mse"  # "binary_crossentropy"
    }

    blackbox = BlackboxWrapper(blackbox_loader("cbf_knn.joblib"), 2, 1)
    blackbox = blackbox_loader("ArrowHead_rocket.joblib")

    encoder, decoder, discriminator, autoencoder = build_aaep(
        input_shape, latent_dim, autoencoder_kwargs, discriminator_kwargs
    )

    # hist = autoencoder.fit(X_exp_train, epochs=10, batch_size=32)

    from late.utils.utils import plot_grouped_history, reconstruction_accuracy_vae, reconstruction_accuracy

    hist = autoencoder.fit(X_exp_train, epochs=4000, validation_split=0.2)  # validation_data=(X_exp_val,))
    plot_grouped_history(hist.history)
    plot_latent_space(encoder.predict(X_train), y_train)
    i = 0
    plt.plot(X_train[i].ravel())
    plt.plot(autoencoder.predict(X_train[i:i + 1]).ravel())
    plt.show()

    plot_reconstruction_vae(X_test[:5], encoder, decoder)

    plot_latent_space_matrix(encoder.predict(X_train))
    plot_latent_space_matrix(encoder.predict(X_train), y_train)

    print(reconstruction_accuracy(X_test, encoder, decoder, blackbox))
