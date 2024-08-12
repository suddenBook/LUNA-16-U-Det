from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
import tensorflow_addons as tfa

import sys
sys.path.extend(['./','../','../Model_Helpers/','../Data_Loader/','../Custom_Functions/'])

def UNet(input_shape=(512, 512, 1), use_mish=True):
    inputs = Input(input_shape)
    activation = tfa.activations.mish if use_mish else 'relu'

    conv1 = Conv2D(64, 3, activation=activation, padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation=activation, padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation=activation, padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation=activation, padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation=activation, padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation=activation, padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation=activation, padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation=activation, padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation=activation, padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation=activation, padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(512, 2, strides=2, padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(512, 3, activation=activation, padding='same')(up6)
    conv6 = Conv2D(512, 3, activation=activation, padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(256, 2, strides=2, padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(256, 3, activation=activation, padding='same')(up7)
    conv7 = Conv2D(256, 3, activation=activation, padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(128, 2, strides=2, padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(128, 3, activation=activation, padding='same')(up8)
    conv8 = Conv2D(128, 3, activation=activation, padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(64, 2, strides=2, padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(64, 3, activation=activation, padding='same')(up9)
    conv9 = Conv2D(64, 3, activation=activation, padding='same')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])