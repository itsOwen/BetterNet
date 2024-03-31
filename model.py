import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Activation, BatchNormalization, UpSampling2D,
                                     Input, Concatenate, Add, Dropout, GlobalAveragePooling2D,
                                     Reshape, Dense, multiply, Resizing)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.layers import GlobalMaxPooling2D, Conv2D
from tensorflow.keras.regularizers import l1_l2 # Regularizers

def channel_attention_module(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    shared_dense_one = Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_dense_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    attention_feature = Add()([avg_pool, max_pool])
    attention_feature = Activation('sigmoid')(attention_feature)

    return multiply([input_feature, attention_feature])

def spatial_attention_module(input_feature):
    kernel_size = 7

    avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    attention_feature = Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)   

    return multiply([input_feature, attention_feature])

def cbam_module(input_feature, ratio=8):
    channel_attention = channel_attention_module(input_feature, ratio)
    spatial_attention = spatial_attention_module(channel_attention)
    return spatial_attention

def squeeze_excitation_module(input_feature, ratio=16):
    channel_axis = -1
    filters = input_feature.shape[channel_axis]
    se_shape = (1, 1, filters)

    squeeze = GlobalAveragePooling2D()(input_feature)
    squeeze = Reshape(se_shape)(squeeze)
    excitation = Dense(filters // ratio, activation='relu')(squeeze)
    excitation = Dense(filters, activation='sigmoid')(excitation)

    scale = multiply([input_feature, excitation])
    return scale

def residual_block(input_feature, num_filters, dropout_rate=0.5):
    x = input_feature

    x = Conv2D(num_filters//4, (1, 1), padding="same", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters//4, (3, 3), padding="same", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3), padding="same", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(num_filters, (1, 1), padding="same", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(input_feature)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    x = squeeze_excitation_module(x)

    x = Dropout(dropout_rate)(x)
    return x

def decoder_block(input_feature, skip_connection, num_filters, dropout_rate=0.5):
    x = UpSampling2D((2, 2), interpolation='bilinear')(input_feature)
    skip_connection = Resizing(x.shape[1], x.shape[2], interpolation='bilinear')(skip_connection)
    x = Concatenate()([x, skip_connection])

    x = residual_block(x, num_filters, dropout_rate=dropout_rate)

    x = cbam_module(x)
    return x

def BetterNet(input_shape, num_classes, dropout_rate=0.5):
    inputs = Input(shape=input_shape, name="input_image")
    encoder = EfficientNetB1(input_tensor=inputs, weights="imagenet", include_top=False)

    skip_connection_names = [
        'input_image',
        'block2a_expand_activation',
        'block3a_expand_activation',
        'block4a_expand_activation',
        'block6a_expand_activation'
    ]
    skip_connections = [encoder.get_layer(name).output for name in skip_connection_names]
    x = encoder.output

    decoder_filters = [192, 128, 64, 32, 16]

    for i, filters in enumerate(decoder_filters):
        if i < len(skip_connections) - 1:
            skip_connection = skip_connections[-(i + 2)]
            x = decoder_block(x, skip_connection, filters, dropout_rate=dropout_rate)

    x = UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = Conv2D(num_classes, (1, 1), padding="same")(x)
    output = Activation("sigmoid", name="output_image")(x)

    model = Model(inputs, outputs=output, name="BetterNet")
    return model

if __name__ == "__main__":
    params = {
        "img_height": 224,
        "img_width": 224,
        "img_channels": 3,
        "mask_channels": 1
    }

    model = BetterNet(input_shape=(params["img_height"], params["img_width"], params["img_channels"]), 
                      num_classes=params["mask_channels"], 
                      dropout_rate=0.5)
    model.summary()