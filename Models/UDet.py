import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model  
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, UpSampling2D
from BiFPN import build_BiFPN

import sys
sys.path.extend(['./','../','../Model_Helpers/','../Data_Loader/','../Custom_Functions/'])

def UDet(input_shape=(512, 512, 1), use_mish=True):
    inputs = Input(input_shape)
    activation = tfa.activations.mish if use_mish else "relu"

    conv1 = Conv2D(64, 3, activation=activation, padding="same")(inputs)
    conv1 = Conv2D(64, 3, activation=activation, padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation=activation, padding="same")(pool1)
    conv2 = Conv2D(128, 3, activation=activation, padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation=activation, padding="same")(pool2)  
    conv3 = Conv2D(256, 3, activation=activation, padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation=activation, padding="same")(pool3)
    conv4 = Conv2D(512, 3, activation=activation, padding="same")(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation=activation, padding="same")(pool4)
    conv5 = Conv2D(1024, 3, activation=activation, padding="same")(conv5)

    channels = [64, 128, 256, 512, 1024]
    P3_out, P4_out, P5_out, P6_out, P7_out = build_BiFPN([conv1, conv2, conv3, conv4, conv5], channels[0], 1)

    # upsampled_conv5 = UpSampling2D(size=(8, 8))(conv5)
    up6 = concatenate([Conv2DTranspose(512, 2, strides=2, padding="same")(conv5), P6_out], axis=3)
    conv6 = Conv2D(512, 3, activation=activation, padding="same")(up6)
    conv6 = Conv2D(512, 3, activation=activation, padding="same")(conv6)

    up7 = concatenate([Conv2DTranspose(256, 2, strides=2, padding="same")(conv6), P5_out], axis=3)
    conv7 = Conv2D(256, 3, activation=activation, padding="same")(up7)  
    conv7 = Conv2D(256, 3, activation=activation, padding="same")(conv7)

    up8 = concatenate([Conv2DTranspose(128, 2, strides=2, padding="same")(conv7), P4_out], axis=3)
    conv8 = Conv2D(128, 3, activation=activation, padding="same")(up8)
    conv8 = Conv2D(128, 3, activation=activation, padding="same")(conv8)

    up9 = concatenate([Conv2DTranspose(64, 2, strides=2, padding="same")(conv8), P3_out], axis=3)
    conv9 = Conv2D(64, 3, activation=activation, padding="same")(up9)
    conv9 = Conv2D(64, 3, activation=activation, padding="same")(conv9)

    conv10 = Conv2D(1, 1, activation="sigmoid")(conv9)

    return Model(inputs=[inputs], outputs=[conv10])