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

def visualize_feature_map(feature_maps, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for layer_idx, feature_map in enumerate(feature_maps):
        if len(feature_map.shape) == 4:
            num_features = feature_map.shape[-1]
            size = feature_map.shape[1]

            for i in range(num_features):
                x = feature_map[0, :, :, i]
                x -= x.mean()
                std_dev = x.std()
                if std_dev == 0:
                    std_dev = 1e-8
                x /= std_dev
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')

                plt.figure(figsize=(5, 5))
                plt.imshow(x, aspect='auto', cmap='gray')
                plt.axis('off')
                save_path = os.path.join(output_folder, f'layer_{layer_idx}_feature_{i}.png')
                plt.savefig(save_path)
                plt.close()

def main():
    base_model = UDet()
    model = modify_model_for_visualization(base_model)
    
    image_path = '../Data/results/udet/mish/split_1/raw_output/389_raw_output.mhd'
    itk_image = sitk.ReadImage(image_path)
    input_data = sitk.GetArrayFromImage(itk_image)
    
    input_data = np.transpose(input_data, (1, 2, 0))
    input_data = np.expand_dims(input_data, axis=0)

    outputs = model.predict(input_data)

    visualize_feature_map(outputs, 'feature_map_slices')

if __name__ == "__main__":
    main()