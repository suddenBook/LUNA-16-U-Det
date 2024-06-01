from functools import reduce
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import models

import sys
sys.path.extend(['./','../','../Model_Helpers/','../Data_Loader/','../Custom_Functions/'])

class BatchNormalization(layers.BatchNormalization):
    """
    Identical to keras.layers.BatchNormalization, but adds the option to freeze parameters.
    """
    def __init__(self, freeze, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze = freeze
        self.trainable = not self.freeze

    def call(self, inputs, training=None, **kwargs):
        if not training:
            return super().call(inputs, training=False)
        else:
            return super().call(inputs, training=(not self.freeze))

    def get_config(self):
        config = super().get_config()
        config.update({"freeze": self.freeze})
        return config

def ConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
    def apply(x):
        x = layers.Conv2D(
            num_channels, kernel_size, strides=strides, padding="same", use_bias=False, name=f"{name}_conv")(x)
        x = BatchNormalization(freeze=freeze_bn, name=f"{name}_bn")(x)
        x = layers.ReLU(name=f"{name}_relu")(x)
        return x
    return apply

def DepthwiseConvBlock(kernel_size, strides, name, freeze_bn=False):
    def apply(x):
        x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding="same", use_bias=False, name=f"{name}_dconv")(x)
        x = BatchNormalization(freeze=freeze_bn, name=f"{name}_bn")(x)
        x = layers.ReLU(name=f"{name}_relu")(x)
        return x
    return apply
        
def build_BiFPN(features, num_channels, id, freeze_bn=False):
    if id == 0:
        _, _, C3, C4, C5 = features
        P3_in = ConvBlock(num_channels, 1, 1, freeze_bn=freeze_bn, name=f"BiFPN_{id}_P3")(C3)
        P4_in = ConvBlock(num_channels, 1, 1, freeze_bn=freeze_bn, name=f"BiFPN_{id}_P4")(C4)
        P5_in = ConvBlock(num_channels, 1, 1, freeze_bn=freeze_bn, name=f"BiFPN_{id}_P5")(C5)
        P6_in = ConvBlock(num_channels, 3, 2, freeze_bn=freeze_bn, name=f"BiFPN_{id}_P6")(C5)
        P7_in = ConvBlock(num_channels, 3, 2, freeze_bn=freeze_bn, name=f"BiFPN_{id}_P7")(P6_in)
    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features
        P3_in = ConvBlock(num_channels, 1, 1, freeze_bn=freeze_bn, name=f"BiFPN_{id}_P3")(P3_in)
        P4_in = ConvBlock(num_channels, 1, 1, freeze_bn=freeze_bn, name=f"BiFPN_{id}_P4")(P4_in)
        P5_in = ConvBlock(num_channels, 1, 1, freeze_bn=freeze_bn, name=f"BiFPN_{id}_P5")(P5_in)
        P6_in = ConvBlock(num_channels, 1, 1, freeze_bn=freeze_bn, name=f"BiFPN_{id}_P6")(P6_in)
        P7_in = ConvBlock(num_channels, 1, 1, freeze_bn=freeze_bn, name=f"BiFPN_{id}_P7")(P7_in)

    # Upsampling path
    P7_U = layers.UpSampling2D()(P7_in)
    P6_td = layers.Add()([P7_U, P6_in])
    P6_td = DepthwiseConvBlock(3, 1, freeze_bn=freeze_bn, name=f"BiFPN_{id}_U_P6")(P6_td)
    P6_U = layers.UpSampling2D()(P6_td)
    P5_td = layers.Add()([P6_U, P5_in])
    P5_td = DepthwiseConvBlock(3, 1, freeze_bn=freeze_bn, name=f"BiFPN_{id}_U_P5")(P5_td)
    P5_U = layers.UpSampling2D()(P5_td)
    P4_td = layers.Add()([P5_U, P4_in])
    P4_td = DepthwiseConvBlock(3, 1, freeze_bn=freeze_bn, name=f"BiFPN_{id}_U_P4")(P4_td)
    P4_U = layers.UpSampling2D()(P4_td)
    P3_out = layers.Add()([P4_U, P3_in])
    P3_out = DepthwiseConvBlock(3, 1, freeze_bn=freeze_bn, name=f"BiFPN_{id}_U_P3")(P3_out)

    # Downsampling path
    P3_D = layers.MaxPooling2D(strides=(2, 2))(P3_out)
    P4_out = layers.Add()([P3_D, P4_td, P4_in])
    P4_out = DepthwiseConvBlock(3, 1, freeze_bn=freeze_bn, name=f"BiFPN_{id}_D_P4")(P4_out)
    P4_D = layers.MaxPooling2D(strides=(2, 2))(P4_out)
    P5_out = layers.Add()([P4_D, P5_td, P5_in])
    P5_out = DepthwiseConvBlock(3, 1, freeze_bn=freeze_bn, name=f"BiFPN_{id}_D_P5")(P5_out)
    P5_D = layers.MaxPooling2D(strides=(2, 2))(P5_out)
    P6_out = layers.Add()([P5_D, P6_td, P6_in])
    P6_out = DepthwiseConvBlock(3, 1, freeze_bn=freeze_bn, name=f"BiFPN_{id}_D_P6")(P6_out)
    P6_D = layers.MaxPooling2D(strides=(2, 2))(P6_out)
    P7_out = layers.Add()([P6_D, P7_in])
    P7_out = DepthwiseConvBlock(3, 1, freeze_bn=freeze_bn, name=f"BiFPN_{id}_D_P7")(P7_out)

    return P3_out, P4_out, P5_out, P6_out, P7_out