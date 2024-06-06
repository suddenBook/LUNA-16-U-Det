We can categorize all layers into the following groups.

1. Input Layer:
`input_1`

2. Convolutional Layers:
`conv2d`, `conv2d_1`, `conv2d_2`, `conv2d_3`, `conv2d_4`, `conv2d_5`, `conv2d_6`, `conv2d_7`, `conv2d_8`, `conv2d_9`, `conv2d_10`, `conv2d_11`, `conv2d_12`, `conv2d_13`, `conv2d_14`, `conv2d_15`, `conv2d_16`, `conv2d_17`, `conv2d_18`

3. Pooling Layers:
`max_pooling2d`, `max_pooling2d_1`, `max_pooling2d_2`, `max_pooling2d_3`, `max_pooling2d_4`, `max_pooling2d_5`, `max_pooling2d_6`

4. Dropout Layer:
`dropout`

5. BiFPN Layers:
`BiFPN_1_P7_conv`, `BiFPN_1_P7_bn`, `BiFPN_1_P6_conv`, `BiFPN_1_P7_relu`, `BiFPN_1_P6_bn`, `BiFPN_1_P6_relu`, `BiFPN_1_U_P6_dconv`, `BiFPN_1_U_P6_bn`, `BiFPN_1_P5_conv`, `BiFPN_1_U_P6_relu`, `BiFPN_1_P5_bn`, `BiFPN_1_P5_relu`, `BiFPN_1_U_P5_dconv`, `BiFPN_1_U_P5_bn`, `BiFPN_1_P4_conv`, `BiFPN_1_U_P5_relu`, `BiFPN_1_P4_bn`, `BiFPN_1_P4_relu`, `BiFPN_1_U_P4_dconv`, `BiFPN_1_U_P4_bn`, `BiFPN_1_P3_conv`, `BiFPN_1_U_P4_relu`, `BiFPN_1_P3_bn`, `BiFPN_1_P3_relu`, `BiFPN_1_U_P3_dconv`, `BiFPN_1_U_P3_bn`, `BiFPN_1_U_P3_relu`, `BiFPN_1_D_P4_dconv`, `BiFPN_1_D_P4_bn`, `BiFPN_1_D_P4_relu`, `BiFPN_1_D_P5_dconv`, `BiFPN_1_D_P5_bn`, `BiFPN_1_D_P5_relu`, `BiFPN_1_D_P6_dconv`, `BiFPN_1_D_P6_bn`, `BiFPN_1_D_P6_relu`

6. Upsampling Layers:
`up_sampling2d`, `up_sampling2d_1`, `up_sampling2d_2`, `up_sampling2d_3`

7. Addition Layers:
`add`, `add_1`, `add_2`, `add_3`, `add_4`, `add_5`, `add_6`

8. Transposed Convolutional Layers:
`conv2d_transpose`, `conv2d_transpose_1`, `conv2d_transpose_2`, `conv2d_transpose_3`

9. Concatenation Layers:
`concatenate`, `concatenate_1`, `concatenate_2`, `concatenate_3`

To create a feature map, we can extract the following layers.

1. The encoder captures low-level features such as edges and textures:
   - `conv2d_1` (index: 1)
   - `conv2d_3` (index: 5)
   - `conv2d_5` (index: 7)
   - `conv2d_7` (index: 11)
   - `conv2d_9` (index: 15)

2. The Bi-FPN module fuses multi-scale features to enhance the representation of nodule regions:
   - `BiFPN_1_P3_relu` (index: 44)
   - `BiFPN_1_P4_relu` (index: 38)
   - `BiFPN_1_P5_relu` (index: 31)
   - `BiFPN_1_P6_relu` (index: 24)
   - `BiFPN_1_P7_relu` (index: 19)

3. The decoder gradually recovers the spatial details and generates the final segmentation mask:
   - `concatenate` (index: 70)
   - `concatenate_1` (index: 73)
   - `concatenate_2` (index: 76)
   - `concatenate_3` (index: 79)
   - `conv2d_18` (index: 82)

The array of indices for the layers is as follows.

```python
layer_indices = [1, 5, 7, 11, 15, 19, 24, 31, 38, 44, 70, 73, 76, 79, 82]
```