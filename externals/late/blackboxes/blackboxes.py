from tensorflow import keras


def build_resnet(n_timesteps, n_features, n_dimensions=1):
    n_feature_maps = 64

    input_layer = keras.layers.Input((n_timesteps, n_dimensions))

    # BLOCK 1 

    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

    # expand channels for the sum 
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)

    # BLOCK 2 

    conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

    # expand channels for the sum 
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)

    # BLOCK 3 

    conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_2)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

    # no need to expand channels because they are equal 
    shortcut_y = keras.layers.normalization.BatchNormalization()(output_block_2)

    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    output_block_3 = keras.layers.Activation('relu')(output_block_3)

    # FINAL 

    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

    output_layer = keras.layers.Dense(n_features, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    return model

  
def build_simple_CNN(n_timesteps, n_outputs):
    model = keras.models.Sequential()
    
    model.add(keras.layers.Conv1D(filters=16, kernel_size=8, activation='relu', input_shape=(n_timesteps,1)))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    
    model.add(keras.layers.Dropout(0.3))
    
    model.add(keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    
    model.add(keras.layers.Dropout(0.3))
    
    model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    
    model.add(keras.layers.Dropout(0.3))
    
    model.add(keras.layers.GlobalAveragePooling1D())
    
    #model.add(Dense(30, activation='relu'))
    model.add(keras.layers.Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


def build_resnet_dropout(n_timesteps, n_features):  
    n_feature_maps = 64

    input_layer = keras.layers.Input((n_timesteps, 1))

    # BLOCK 1 

    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

    # expand channels for the sum 
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)
    output_block_1 = keras.layers.Dropout(0.3)(output_block_1)

    # BLOCK 2 

    conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

    # expand channels for the sum 
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)
    output_block_2 = keras.layers.Dropout(0.3)(output_block_2)

    # BLOCK 3 

    conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_2)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

    # no need to expand channels because they are equal 
    shortcut_y = keras.layers.normalization.BatchNormalization()(output_block_2)

    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    output_block_3 = keras.layers.Activation('relu')(output_block_3)
    output_block_3 = keras.layers.Dropout(0.3)(output_block_3)

    # FINAL 

    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

    output_layer = keras.layers.Dense(n_features, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    return model