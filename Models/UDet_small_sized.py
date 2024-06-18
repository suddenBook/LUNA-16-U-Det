import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, UpSampling2D, add
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Permute, multiply, Activation
from BiFPN import build_BiFPN

def conv_block(inputs, num_filters, kernel_size=3):
    x = Conv2D(num_filters, kernel_size, padding="same")(inputs)
    x = tfa.activations.mish(x)
    x = Conv2D(num_filters, kernel_size, padding="same")(x)
    x = tfa.activations.mish(x)
    return x

def attention_block(input, filters):
    x = Conv2D(filters, kernel_size=1)(input)
    x = tfa.activations.mish(x)
    x = Conv2D(input.shape[-1], kernel_size=1)(x)
    x = Activation('sigmoid')(x)
    x = multiply([input, x])
    return x

def UDet_small_sized(input_shape=(512, 512, 1)):
    inputs = Input(input_shape)

    conv1 = conv_block(inputs, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, 256)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = conv_block(pool4, 512)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = conv_block(pool5, 1024)
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)

    conv7 = conv_block(pool6, 2048)

    P3_out, P4_out, P5_out, P6_out, P7_out = build_BiFPN([conv3, conv4, conv5, conv6, conv7], 128, 1)

    up7 = attention_block(conv7, 256)
    up7 = UpSampling2D(size=(8, 8), interpolation='bilinear')(up7)
    P7_up = UpSampling2D(size=(8, 8), interpolation='bilinear')(P7_out)

    up9 = attention_block(conv3, 128)
    up9 = UpSampling2D(size=(2, 2), interpolation='bilinear')(up9)

    P5_up = UpSampling2D(size=(8, 8), interpolation='bilinear')(P5_out)

    merge9 = concatenate([up9, P5_up, conv2], axis=3)

    up10 = UpSampling2D(size=(2, 2), interpolation='bilinear')(merge9)
    conv10 = Conv2D(64, 3, activation='relu', padding='same')(up10)
    conv11 = Conv2D(64, 3, activation='relu', padding='same')(conv10)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv11)

    return Model(inputs, outputs)
