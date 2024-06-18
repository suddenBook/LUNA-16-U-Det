import os
import csv
import numpy as np
import SimpleITK as sitk
from glob import glob
from sklearn.metrics import confusion_matrix
from collections import defaultdict

import sys
sys.path.extend(['./','../','../../','../../Models/','../../Custom_Functions/','../../Model_Helpers/','../../Data_Loader/','../../Data/'])

def read_csv(filename):
    lines = []
    with open(filename, 'r') as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            lines.append(line)
    return lines

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing

def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates[:3] - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return voxel_coordinates

def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates

def compute_metrics(y_true, y_pred):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    DSC = 2 * TP / (FP + 2 * TP + FN)
    SEN = TP / (TP + FN)
    PPV = TP / (TP + FP)
    return DSC, SEN, PPV

def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model_name, activation, data_root, annotations_path, results_dir, seriesuid_map):
    annotations = read_csv(annotations_path)[1:]

    # generate {seriesuid: [(coordX, coordY, coordZ, diameter_mm)]} mapping
    ground_truth = defaultdict(list)
    for row in annotations[1:]:
        seriesuid, coordX, coordY, coordZ, diameter_mm = row
        ground_truth[seriesuid].append((float(coordX), float(coordY), float(coordZ), float(diameter_mm)))

    pred_dir = os.path.join(results_dir, model_name, activation, 'split_1', 'final_output')
    metrics = []

    for pred_file in glob(os.path.join(pred_dir, '*.mhd')):
        file_name = os.path.basename(pred_file)
        if 'final_output' not in file_name:
            continue
        file_prefix = file_name.split('_')[0]
        seriesuid = [k for k, v in seriesuid_map.items() if v == file_prefix]
        if not seriesuid:
            continue
        seriesuid = seriesuid[0]

        pred_img, origin, spacing = load_itk_image(pred_file)
        pred_img = (pred_img > 0).astype(np.uint8)

        gt_coords = np.array([world_2_voxel(coord[:3], origin, spacing) for coord in ground_truth[seriesuid]])
        y_true, y_pred = [], []

        for coord in gt_coords:
            x, y, z = coord.astype(int)
            if x < 0 or x >= pred_img.shape[2] or y < 0 or y >= pred_img.shape[1] or z < 0 or z >= pred_img.shape[0]:
                continue
            y_true.append(1)
            y_pred.append(pred_img[z, y, x])

        if len(y_true) > 0:
            DSC, SEN, PPV = compute_metrics(y_true, y_pred)
            metrics.append((DSC, SEN, PPV))

    if model_name == 'bifpn':
        from Models.Encoder_BiFPN import BIFPN
        model = BIFPN(input_shape=(512, 512, 5))
    elif model_name == 'unet':
        from Models.UNet import UNet
        model = UNet(input_shape=(512, 512, 5))
    elif model_name == 'udet':
        from Models.UDet import UDet
        model = UDet(input_shape=(512, 512, 5))
    elif model_name == 'udet_small_sized':
        from Models.UDet_small_sized import UDet_small_sized
        model = UDet_small_sized(input_shape=(512, 512, 5))
    else:
        raise ValueError(f'Unknown model: {model_name}')

    num_params = sum(p.numpy().size for p in model.trainable_weights)
    num_layers = len(model.layers)

    if metrics:
        metrics = np.array(metrics)
        mean_metrics = metrics.mean(axis=0)
        DSC, SEN, PPV = mean_metrics
    else:
        print(f"\nNo valid predictions found for {model_name} with {activation} activation.")
        DSC, SEN, PPV = 0, 0, 0

    print(f'\nResults for {model_name} with {activation} activation:')
    print(f'DSC: {DSC:.4f}')
    print(f'SEN: {SEN:.4f}')
    print(f'PPV: {PPV:.4f}')

if __name__ == '__main__':
    data_root = '../Data/'
    annotations_path = './annotations/annotations.csv'
    results_dir = os.path.join(data_root, 'results')
    seriesuid_map_path = os.path.join(data_root, 'seriesuid_mapping.csv')
    seriesuid_map = dict(read_csv(seriesuid_map_path))

    for model_name in ['unet', 'udet', 'bifpn']:
        for activation in ['mish', 'relu']:
            evaluate(model_name, activation, data_root, annotations_path, results_dir, seriesuid_map)