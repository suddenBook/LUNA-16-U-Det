from __future__ import print_function

import sys
sys.path.extend(['./','./Models/','./Model_Helpers/','./Data_Loader/','./Custom_Functions/'])

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
plt.ioff()
from tensorflow.python.keras import backend as K
K.set_image_data_format("channels_last")
import os
import csv
from tqdm import tqdm
import numpy as np
import scipy.ndimage.morphology
from skimage import measure, filters
import SimpleITK as sitk
from Model_Helpers.metrics import dc, jc, assd
from Data_Loader.load_3D_data import generate_test_batches
import logging
from scipy.spatial import distance

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_iou(bbox1, bbox2):
    """
    Calculate the IoU between two bounding boxes.

    IoU is a measure used in object detection to quantify the percent overlap between two bounding boxes.
    It is defined as the area of the intersection divided by the area of the union of the two bounding boxes.
    This function is used to determine how much one bounding box overlaps with another, which is crucial for tasks such as object detection, tracking, and in scenarios where predictions from different models need to be compared or combined.

    Parameters:
    - bbox1 (tuple): The (x1, y1, x2, y2) coordinates of the first bounding box.
    - bbox2 (tuple): The (x1, y1, x2, y2) coordinates of the second bounding box.

    Returns:
    - float: The IoU ratio, a value between 0 and 1, where 0 means no overlap and 1 means perfect overlap.

    The function works by first calculating the coordinates of the intersection of bbox1 and bbox2.
    If the bounding boxes do not overlap, the intersection area is zero. The union is calculated as the sum of the individual areas of the bounding boxes minus the intersection area. The function returns the intersection area divided by the union area, providing a measure of the overlap between the two bounding boxes.
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def get_bounding_box(binary_mask):
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return x1, y1, x2, y2

def calculate_confidence(prediction, size):
    return np.mean(prediction) * (1 + np.log(size))
    # size is used as a logarithmic factor


def threshold_mask(raw_output, threshold):
    if threshold == 0:
        try:
            threshold = filters.threshold_otsu(raw_output)
        except:
            threshold = 0.5
            
    print(f"\tThreshold: {threshold}")
    
    raw_output[raw_output > threshold] = 1
    raw_output[raw_output < 1] = 0
    
    all_labels = measure.label(raw_output)
    props = measure.regionprops(all_labels)
    props.sort(key=lambda x: x.area, reverse=True)
    thresholded_mask = np.zeros(raw_output.shape)
    
    if len(props) >= 2:
        if props[0].area / props[1].area > 5:
            thresholded_mask[all_labels == props[0].label] = 1
        else:
            thresholded_mask[all_labels == props[0].label] = 1
            thresholded_mask[all_labels == props[1].label] = 1
    elif len(props):
        thresholded_mask[all_labels == props[0].label] = 1
        
    thresholded_mask = scipy.ndimage.morphology.binary_fill_holes(thresholded_mask).astype(np.uint8)
    return thresholded_mask

def predict(model, img, args):
    img_data = np.load(os.path.join(args.data_root_dir, "imgs", "images_" + img[0])).T
    img_data = img_data[np.newaxis, :, :]
    num_slices = img_data.shape[0]
    
    output_array = model.predict(
        generate_test_batches(
            args.data_root_dir, 
            [img], 
            (img_data.shape[1],img_data.shape[2],args.slices),
            batchSize=args.batch_size,
            numSlices=args.slices,
            subSampAmt=0,
            stride=1
        ),
        steps = num_slices,
        max_queue_size = 1,
        workers = 1, 
        use_multiprocessing = False,
        verbose = 1,
    )
    output = output_array[:, :, :, 0] if args.net.find("caps") == -1 else output_array[0][:, :, :, 0]
    return output

def save_output(output, output_dir, img, args):
    output_img = sitk.GetImageFromArray(output)
    print(f"Shape of numpy array: {output.shape}")
    print("Segmenting output")
    output_bin = threshold_mask(output, args.thresh_level)
    output_mask = sitk.GetImageFromArray(output_bin)
    
    print("Saving output")  
    writer = sitk.ImageFileWriter()
    writer.SetFileName(os.path.join(output_dir, "raw_output", img[0][:-4] + "_raw_output" + img[0][-4:].replace(".npy", ".mhd")))
    writer.Execute(output_img)
    writer.SetFileName(os.path.join(output_dir, "final_output", img[0][:-4] + "_final_output" + img[0][-4:].replace(".npy", ".mhd")))  
    writer.Execute(output_mask)

def load_gt_mask(args, img):
    gt_data = np.load(os.path.join(args.data_root_dir, "masks", "masks_" + img[0])).T
    gt_data = gt_data[np.newaxis, :, :]
    return gt_data

def save_qual_fig(output, gt_data, output_dir, img, img_data):
    print("Creating qualitative figure for quick reference")
    f, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    slices = [img_data.shape[0] // 3, img_data.shape[0] // 2, img_data.shape[0] // 2 + img_data.shape[0] // 4]
    for i, si in enumerate(slices):
        ax[i].imshow(img_data[si, :, :], alpha=1, cmap="gray")
        ax[i].imshow(output[si, :, :], alpha=0.5, cmap="Blues")
        ax[i].imshow(gt_data[si, :, :], alpha=0.2, cmap="Reds")
        ax[i].set_title(f"Slice {si}/{img_data.shape[0]}")
        ax[i].axis("off")
        
    fig = plt.gcf()
    fig.suptitle(img[0][:-4])  
    plt.savefig(os.path.join(output_dir, "qual_figs", img[0][:-4] + "_qual_fig" + ".png"), bbox_inches="tight")
    plt.close()

def compute_metrics(output_bin, gt_data, img, sitk_img, args):
    gt_data = gt_data.squeeze()
    row = [img[0][:-4]]
    if args.compute_dice:
        print("Computing Dice")
        dice = dc(output_bin, gt_data)
        print(f"\tDice: {dice}")
        row.append(dice)
    if args.compute_jaccard:
        print("Computing Jaccard")  
        jaccard = jc(output_bin, gt_data)  
        print(f"\tJaccard: {jaccard}")
        row.append(jaccard)
    if args.compute_assd:  
        print("Computing ASSD")
        assd_ = assd(output_bin, gt_data, voxelspacing=sitk_img.GetSpacing(), connectivity=1)
        print(f"\tASSD: {assd_}")  
        row.append(assd_)
    return row

def calculate_nodule_size(nodule_mask):
    labeled_mask, num_components = measure.label(nodule_mask, return_num=True)
    if num_components > 0:
        largest_component = labeled_mask == np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1
        return np.sum(largest_component)
    else:
        return 0

def test_udet_mixed(args, test_list, model_list, net_input_shape, udet_weights_path, udet_small_sized_weights_path):
    udet_model = model_list[0]
    udet_small_sized_model = model_list[1]

    try:
        udet_model.load_weights(udet_weights_path)
        udet_small_sized_model.load_weights(udet_small_sized_weights_path)
    except:
        print("Unable to find weights. Testing with random weights.")

    output_dir = os.path.join(args.data_root_dir, "results", "udet_mixed", "split_" + str(args.split_num))
    os.makedirs(output_dir, exist_ok=True)

    raw_out_dir = os.path.join(output_dir, "raw_output")
    fin_out_dir = os.path.join(output_dir, "final_output")
    fig_out_dir = os.path.join(output_dir, "qual_figs")

    for out_dir in [raw_out_dir, fin_out_dir, fig_out_dir]:
        os.makedirs(out_dir, exist_ok=True)
    
    size_threshold = 5000
    iou_threshold = 0.5

    for img in tqdm(test_list):
        udet_output = predict(udet_model, img, args)
        udet_small_sized_output = predict(udet_small_sized_model, img, args)

        # apply thresholding to binarize the outputs
        udet_bin = threshold_mask(udet_output, args.thresh_level)
        udet_small_sized_bin = threshold_mask(udet_small_sized_output, args.thresh_level)

        mixed_output = np.zeros_like(udet_output)

        # label connected components in both binarized outputs
        udet_labels = np.squeeze(udet_bin)
        udet_small_sized_labels = np.squeeze(udet_small_sized_bin)

        udet_labels, udet_num = measure.label(udet_labels, return_num=True)
        udet_small_sized_labels, udet_small_sized_num = measure.label(udet_small_sized_labels, return_num=True)

        nodule_info = []

        for i in range(1, udet_num + 1):
            nodule_mask = (udet_labels == i)
            nodule_size = np.sum(nodule_mask)
            bbox = get_bounding_box(nodule_mask)
            confidence = calculate_confidence(udet_output[0][nodule_mask], nodule_size)
            nodule_info.append(('udet', i, bbox, nodule_size, confidence))

        for i in range(1, udet_small_sized_num + 1):
            nodule_mask = (udet_small_sized_labels == i)
            nodule_size = np.sum(nodule_mask)
            bbox = get_bounding_box(nodule_mask)
            confidence = calculate_confidence(udet_small_sized_output[0][nodule_mask], nodule_size)
            nodule_info.append(('udet_small', i, bbox, nodule_size, confidence))
        
        processed_nodules = set()

        # resolve conflicts between overlapping nodules detected by different models
        for i, (model1, label1, bbox1, size1, conf1) in enumerate(nodule_info):
            if i in processed_nodules:
                continue

            conflicts = []

            for j, (model2, label2, bbox2, size2, conf2) in enumerate(nodule_info[i+1:], start=i+1):
                if calculate_iou(bbox1, bbox2) > iou_threshold:
                    conflicts.append((j, model2, label2, bbox2, size2, conf2))
                    processed_nodules.add(j)

            if not conflicts:
                if (model1 == 'udet' and size1 > size_threshold) or (model1 == 'udet_small' and size1 <= size_threshold):
                    if model1 == 'udet':
                        mixed_output[0][udet_labels == label1] = 1
                    else:
                        mixed_output[0][udet_small_sized_labels == label1] = 1
                    logger.info(f"Using {model1} prediction for nodule {i}. Size: {size1}, Confidence: {conf1:.4f}")
            else:
                all_predictions = [(i, model1, label1, bbox1, size1, conf1)] + conflicts
                best_prediction = max(all_predictions, key=lambda x: x[5])  # x[5] is confidence
                best_index, best_model, best_label, best_bbox, best_size, best_conf = best_prediction
                
                if (best_model == 'udet' and best_size > size_threshold) or (best_model == 'udet_small' and best_size <= size_threshold):
                    if best_model == 'udet':
                        mixed_output[0][udet_labels == best_label] = 1
                    else:
                        mixed_output[0][udet_small_sized_labels == best_label] = 1
                    logger.info(f"Resolved conflict for nodule {i}. Using {best_model} prediction. Size: {best_size}, Confidence: {best_conf:.4f}")
                else:
                    logger.info(f"Discarded conflicting predictions for nodule {i}. Best model: {best_model}, Size: {best_size}, Confidence: {best_conf:.4f}")

        save_output(mixed_output, output_dir, img, args)

    print("U-Det Mixed method testing completed.")


def test(args, test_list, model_list, net_input_shape):
    weights_path = os.path.join(args.check_dir, args.output_name + "_model_" + args.time + ".hdf5") if args.weights_path == "" else args.weights_path
    output_dir = os.path.join(args.data_root_dir, "results", args.net, args.activation, "split_" + str(args.split_num))
    
    raw_out_dir = os.path.join(output_dir, "raw_output")
    fin_out_dir = os.path.join(output_dir, "final_output")
    fig_out_dir = os.path.join(output_dir, "qual_figs")

    for out_dir in [raw_out_dir, fin_out_dir, fig_out_dir]:
        os.makedirs(out_dir, exist_ok=True)

    eval_model = model_list[1] if len(model_list) > 1 else model_list[0]
    try:
        eval_model.load_weights(weights_path)
    except:
        print("Unable to find weights. Testing with random weights.")
    eval_model.summary(positions=[0.38, 0.65, 0.75, 1.0])

    outfile = ""

    if args.compute_dice:
        dice_arr = np.zeros((len(test_list)))
        outfile += "dice_"
    if args.compute_jaccard:
        jacc_arr = np.zeros((len(test_list))) 
        outfile += "jacc_"
    if args.compute_assd:
        assd_arr = np.zeros((len(test_list)))
        outfile += "assd_"

    print("Testing... This will take some time...")

    with open(os.path.join(output_dir, args.save_prefix + outfile + "scores.csv"), "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        header = ["Scan Name"]
        if args.compute_dice:
            header.append("Dice Coefficient")
        if args.compute_jaccard:
            header.append("Jaccard Index")
        if args.compute_assd:
            header.append("Average Symmetric Surface Distance")
        writer.writerow(header)

        for i, img in enumerate(tqdm(test_list)):
            output = predict(eval_model, img, args)
            img_data = np.load(os.path.join(args.data_root_dir, "imgs", "images_" + img[0])).T
            img_data = img_data[np.newaxis, :, :]
            sitk_img = sitk.GetImageFromArray(img_data)
            
            save_output(output, output_dir, img, args)
            gt_data = load_gt_mask(args, img)
            save_qual_fig(output, gt_data, output_dir, img, img_data)
            
            row = compute_metrics(output, gt_data, img, sitk_img, args)
            writer.writerow(row)

            if args.compute_dice:
                dice_arr[i] = row[1]
            if args.compute_jaccard:
                jacc_arr[i] = row[2]
            if args.compute_assd:
                assd_arr[i] = row[3]

        row = ["Average Scores"]
        if args.compute_dice:
            row.append(np.mean(dice_arr))
        if args.compute_jaccard:
            row.append(np.mean(jacc_arr))
        if args.compute_assd:  
            row.append(np.mean(assd_arr))
        writer.writerow(row)
    print("Done.")