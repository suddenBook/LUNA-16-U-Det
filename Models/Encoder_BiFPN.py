import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D
from BiFPN import build_BiFPN

import sys
sys.path.extend(['./','../','../Model_Helpers/','../Data_Loader/','../Custom_Functions/'])

def BIFPN(input_shape=(512, 512, 1), use_mish=True):
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
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation=activation, padding="same")(pool4)
    conv5 = Conv2D(1024, 3, activation=activation, padding="same")(conv5)

    channels = [64, 128, 256, 512, 1024]
    P3_out, P4_out, P5_out, P6_out, P7_out = build_BiFPN([conv1, conv2, conv3, conv4, conv5], channels[0], 1)

    conv6 = Conv2D(64, 3, activation=activation, padding="same")(P3_out)
    conv7 = Conv2D(1, 1, activation="sigmoid")(conv6)

    return Model(inputs=[inputs], outputs=[conv7])