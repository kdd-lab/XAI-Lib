import tensorflow as tf
from tensorflow import keras
from late.autoencoders.tools import repeat_block
import matplotlib.pyplot as plt
import pathlib
from joblib import dump, load
from late.plots.plots import plot_latent_space_matrix, plot_latent_space


def save_model(model, input_shape, latent_dim, autoencoder_kwargs, discriminator_kwargs, path="weights", verbose=False):
    path = pathlib.Path(path)
    model_kwargs = {
        "input_shape": input_shape,
        "latent_dim": latent_dim,
        "autoencoder_kwargs": autoencoder_kwargs,
        "discriminator_kwargs": discriminator_kwargs
    }
    model.save_weights(path.parents[0] / (path.name + ".h5"))
    dump(model_kwargs, path.parents[0] / (path.name + ".joblib"))
    return


def load_model(path="weights", verbose=False):
    path = pathlib.Path(path)
    model_kwargs = load(path.parents[0] / (path.name + ".joblib"))
    encoder, decoder, discriminator, autoencoder = build_aae(
        model_kwargs.get("input_shape"),
        model_kwargs.get("latent_dim"),
        model_kwargs.get("autoencoder_kwargs"),
        model_kwargs.get("discriminator_kwargs"),
        verbose=verbose
    )
    autoencoder.load_weights(path.parents[0] / (path.name + ".h5"))
    return encoder, decoder, autoencoder


def discriminator_loss(real_output, fake_output, loss, loss_weight=1):
    real_loss = loss(tf.ones_like(real_output), real_output)
    fake_loss = loss(tf.zeros_like(fake_output), fake_output)
    total_loss = loss_weight * (real_loss + fake_loss)
    return total_loss


def generator_loss(fake_output, loss, loss_weight=1):
    return loss_weight * loss(tf.ones_like(fake_output), fake_output)


def autoencoder_loss(inputs, reconstruction, loss, loss_weight=1):
    return loss_weight * loss(inputs, reconstruction)


def discriminator_accuracy(y_real, y_fake):
    accuracy = tf.keras.metrics.Accuracy()
    return accuracy(y_real, y_fake)


def build_discriminator(latent_dim, **kwargs):
    discriminator_input = tf.keras.layers.Input(shape=(latent_dim,))
    discriminator_layers = repeat_block(
        discriminator_input,
        "discriminator",
        units=kwargs.get("units"),
        activation=kwargs.get("activation"),
        n_layers=kwargs.get("n_layers")
    )
    discriminator_output = tf.keras.layers.Dense(1, activation="sigmoid")(discriminator_layers)
    discriminator = tf.keras.models.Model(discriminator_input, discriminator_output, name="Discriminator")
    return discriminator


def build_encoder(
        input_shape,
        latent_dim,
        **kwargs
):
    encoder_input = tf.keras.layers.Input(shape=(input_shape))
    encoder_layers = repeat_block(
        encoder_input,
        "encoder",
        filters=kwargs.get("filters"),
        kernel_size=kwargs.get("kernel_size"),
        padding=kwargs.get("padding"),
        activation=kwargs.get("activation"),
        n_layers=kwargs.get("n_layers"),
        pooling=kwargs.get("pooling"),
        batch_normalization=kwargs.get("batch_normalization"),
        n_layers_residual=kwargs.get("n_layers_residual")
    )
    encoder_layers = tf.keras.layers.Conv1D(
        filters=input_shape[1],  # FIXME: or 1?
        kernel_size=1,  # FIXME: maybe different value?
        padding="same")(encoder_layers)
    encoder_layers = tf.keras.layers.Flatten()(encoder_layers)
    encoder_layers = tf.keras.layers.Dense(latent_dim)(encoder_layers)
    encoder_output = encoder_layers
    encoder = tf.keras.models.Model(encoder_input, encoder_output, name="Encoder")
    return encoder


def build_decoder(
        encoder,
        latent_dim,
        **kwargs
):
    decoder_input = tf.keras.layers.Input(shape=(latent_dim,), name='z')
    decoder_layers = decoder_input
    decoder_layers = tf.keras.layers.Dense(encoder.layers[-2].output_shape[1])(decoder_layers)
    decoder_layers = tf.keras.layers.Reshape(encoder.layers[-3].output_shape[1:])(decoder_layers)

    decoder_layers = repeat_block(
        decoder_layers,
        "decoder",
        filters=kwargs.get("filters")[::-1],
        kernel_size=kwargs.get("kernel_size")[::-1],
        padding=kwargs.get("padding")[::-1],
        activation=kwargs.get("activation")[::-1],
        n_layers=kwargs.get("n_layers"),
        pooling=kwargs.get("pooling")[::-1],
        batch_normalization=kwargs.get("batch_normalization"),
        n_layers_residual=kwargs.get("n_layers_residual")
    )
    decoder_output = tf.keras.layers.Conv1D(
        filters=encoder.input_shape[2],
        kernel_size=1,
        padding="same")(decoder_layers)
    decoder = tf.keras.models.Model(decoder_input, decoder_output, name="Decoder")
    return decoder


class AAE(keras.Model):
    def __init__(self, encoder, decoder, discriminator, latent_dim, **kwargs):
        super(AAE, self).__init__(**kwargs)
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
        super(AAE, self).compile()
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
        return self.decoder(self.encoder(inputs))

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
            latent_fake = self.encoder(data, training=True)
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
        latent_fake = self.encoder(data)
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


def build_aae(
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

    decoder = build_decoder(
        encoder,
        latent_dim,
        **autoencoder_kwargs
    )

    discriminator = build_discriminator(
        latent_dim,
        **discriminator_kwargs
    )

    autoencoder = AAE(encoder, decoder, discriminator, latent_dim)
    autoencoder.compile(
        disc_optimizer=discriminator_kwargs.get("optimizer", keras.optimizers.Adam()),
        gen_optimizer=autoencoder_kwargs.get("optimizer", keras.optimizers.Adam()),
        aut_optimizer=autoencoder_kwargs.get("optimizer", keras.optimizers.Adam()),
        disc_loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        gen_loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
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

    return encoder, decoder, discriminator, autoencoder



if __name__ == "__main__":
    from late.datasets.datasets import build_cbf, load_ucr_dataset
    from late.blackboxes.loader import blackbox_loader
    from late.blackboxes.blackbox_wrapper import BlackboxWrapper
    from late.utils.utils import plot_reconstruction_vae
    from late.autoencoders.callbacks import ReduceAAELROnPlateau

    (X_train, y_train, X_val, y_val,
     X_test, y_test, X_exp_train, y_exp_train,
     X_exp_val, y_exp_val, X_exp_test, y_exp_test) = build_cbf(n_samples=600)

    (X_train, y_train, X_val, y_val,
     X_test, y_test, X_exp_train, y_exp_train,
     X_exp_val, y_exp_val, X_exp_test, y_exp_test) = load_ucr_dataset("ECG5000")

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
        "loss": "mse" #"binary_crossentropy"
    }

    reduce_lr = ReduceAAELROnPlateau(monitor='val_aut_loss', factor=0.5, patience=500, min_lr=0.00001, verbose=1)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_aut_loss",
        min_delta=0,
        patience=1000,
        verbose=1,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )

    callbacks = [early_stopping, reduce_lr]

    blackbox = BlackboxWrapper(blackbox_loader("cbf_knn.joblib"), 2, 1)
    blackbox = blackbox_loader("ECG5000_rocket.joblib")

    encoder, decoder, discriminator, autoencoder = build_aae(
        input_shape, latent_dim, autoencoder_kwargs, discriminator_kwargs
    )

    from late.utils.utils import plot_grouped_history, reconstruction_accuracy

    hist = autoencoder.fit(X_exp_train, epochs=8000, callbacks=callbacks, validation_split=0.2)
    # hist = autoencoder.fit(X_exp_train, epochs=2000, callbacks=callbacks, validation_data=(X_exp_val,))
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