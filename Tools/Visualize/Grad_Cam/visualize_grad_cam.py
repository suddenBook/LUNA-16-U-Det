import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.models import Model
import SimpleITK as sitk

import sys
sys.path.extend(['./','../','../../','../../Model_Helpers/','../../Models/','../../Data_Loader/','../../Custom_Functions/'])

from Models.UDet import UDet

def modify_model_for_visualization(base_model):
    layer_outputs = [layer.output for layer in base_model.layers]
    visualization_model = Model(inputs=base_model.input, outputs=layer_outputs)
    return visualization_model

def grad_cam(model, image, layer_name):
    grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[0, 0, 0, 0]
    
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(output, weights)
    
    cam = np.maximum(cam, 0)  # ReLU
    cam /= np.max(cam)  # Normalize
    
    return cam

def visualize_grad_cam(model, image, mask, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    layer_name = 'conv2d_18'
    cam = grad_cam(model, image, layer_name=layer_name)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image[0, :, :, 0], cmap='gray')
    axs[0].set_title('Input Image')
    axs[0].axis('off')
    axs[1].imshow(mask[0, :, :, 0], cmap='gray')
    axs[1].set_title('Ground Truth Mask')
    axs[1].axis('off')
    plt.tight_layout()
    save_path = os.path.join(output_folder, 'grad_cam.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    base_model = UDet()
    base_model.load_weights('../../Data/saved_models/udet/mish/udet_mish_split-0_batch-2_shuff-1_aug-1_loss-w_bce_slic-1_sub--1_strid-1_lr-0.0001_model_2024-05-25-23-12-43.hdf5')
    model = modify_model_for_visualization(base_model)
    
    image_path = '../../Data/results/udet/mish/split_1/raw_output/389_raw_output.mhd'
    mask_path = '../../Data/results/udet/mish/split_1/final_output/389_final_output.mhd'
    
    itk_image = sitk.ReadImage(image_path)
    input_data = sitk.GetArrayFromImage(itk_image)
    input_data = np.transpose(input_data, (1, 2, 0))
    input_data = np.expand_dims(input_data, axis=0)

    itk_mask = sitk.ReadImage(mask_path)
    mask_data = sitk.GetArrayFromImage(itk_mask)
    mask_data = np.transpose(mask_data, (1, 2, 0))
    mask_data = np.expand_dims(mask_data, axis=0)
    
    print("Input data shape:", input_data.shape)
    print("Mask data shape:", mask_data.shape)

    outputs = model.predict(input_data)
    
    visualize_grad_cam(model, input_data, mask_data, 'grad_cam_visualization')

if __name__ == "__main__":
    main()