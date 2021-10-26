from tensorflow import keras


def build_block(
        previous_layer,
        kind,
        filters=None,
        kernel_size=None,
        padding=None,
        activation=None,
        pooling=None,
        units=None,
        batch_normalization=None
):
    if kind == "encoder" or kind == "discriminator_gan":
        layer = keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding)(previous_layer)
        if batch_normalization is not None:
            layer = keras.layers.BatchNormalization()(layer)
        if activation == "leakyrelu":
            layer = keras.layers.LeakyReLU()(layer)
        else:
            layer = keras.layers.Activation(activation)(layer)
        layer = keras.layers.MaxPooling1D(pooling)(layer)
    elif kind == "decoder" or kind == "generator":
        layer = keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding)(previous_layer)
        if batch_normalization is not None:
            layer = keras.layers.BatchNormalization()(layer)
        if activation == "leakyrelu":
            layer = keras.layers.LeakyReLU()(layer)
        else:
            layer = keras.layers.Activation(activation)(layer)
        layer = keras.layers.UpSampling1D(size=pooling)(layer)
    elif kind == "discriminator":
        layer = keras.layers.Dense(units)(previous_layer)
        layer = keras.layers.Activation(activation)(layer)
    else:
        raise Exception("Block kind not valid")
    return layer


def build_residual_block(
        previous_layer,
        kind,
        n_layers,
        filters=None,
        kernel_size=None,
        padding=None,
        activation=None,
        pooling=None,
        units=None,
        batch_normalization=None
):
    if kind == "decoder":
        previous_layer = keras.layers.UpSampling1D(size=pooling)(previous_layer)
    input_layer = previous_layer
    for i in range(n_layers):
        block = build_block(
            previous_layer=previous_layer,
            kind=kind,
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation,
            pooling=1,
            units=units,
            batch_normalization=batch_normalization,
        )
        previous_layer = block
    shortcut = keras.layers.Conv1D(filters=filters, kernel_size=1, padding='same')(input_layer)
    if batch_normalization is not None:
        shortcut = keras.layers.normalization.BatchNormalization()(shortcut)
    output_layer = keras.layers.add([shortcut, block])
    if activation == "leakyrelu":
        output_layer = keras.layers.LeakyReLU()(output_layer)
    else:
        output_layer = keras.layers.Activation(activation)(output_layer)
    if kind == "encoder":
        output_layer = keras.layers.MaxPooling1D(pooling)(output_layer)
    return output_layer


def repeat_block(previous_layer,
                 kind,
                 n_layers,
                 filters=None,
                 kernel_size=None,
                 padding=None,
                 activation=None,
                 pooling=None,
                 units=None,
                 batch_normalization=None,
                 n_layers_residual=None
                 ):
    for i in range(n_layers):
        if n_layers_residual is None:
            block = build_block(
                previous_layer=previous_layer,
                kind=kind,
                filters=filters[i] if filters is not None else None,
                kernel_size=kernel_size[i] if kernel_size is not None else None,
                padding=padding[i] if padding is not None else None,
                activation=activation[i] if activation is not None else None,
                pooling=pooling[i] if pooling is not None else None,
                units=units[i] if units is not None else None,
                batch_normalization=batch_normalization
            )
            previous_layer = block
        else:
            block = build_residual_block(
                previous_layer=previous_layer,
                kind=kind,
                n_layers=n_layers_residual,
                filters=filters[i] if filters is not None else None,
                kernel_size=kernel_size[i] if kernel_size is not None else None,
                padding=padding[i] if padding is not None else None,
                activation=activation[i] if activation is not None else None,
                pooling=pooling[i] if pooling is not None else None,
                units=units[i] if units is not None else None,
                batch_normalization=batch_normalization
            )
            previous_layer = block
    return block



