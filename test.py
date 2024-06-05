from __future__ import print_function

import sys
sys.path.extend(['./','./Models/','./Model_Helpers/','./Data_Loader/','./Custom_Functions/'])

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
plt.ioff()
from tensorflow.keras import backend as K
K.set_image_data_format("channels_last")
import os
import csv
from tqdm import tqdm
import numpy as np
import scipy.ndimage.morphology
from skimage import measure, filters
import SimpleITK as sitk
from metrics import dc, jc, assd
from load_3D_data import generate_test_batches

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    writer.SetFileName(os.path.join(output_dir, img[0][:-4] + "_raw_output" + img[0][-4:].replace(".npy", ".mhd")))
    writer.Execute(output_img)
    writer.SetFileName(os.path.join(output_dir, img[0][:-4] + "_final_output" + img[0][-4:].replace(".npy", ".mhd")))  
    writer.Execute(output_mask)

def load_gt_mask(args, img):
    gt_data = np.load(os.path.join(args.data_root_dir, "masks", "masks_" + img[0])).T
    gt_data = gt_data[np.newaxis, :, :]
    return gt_data

def save_qual_fig(output, gt_data, fig_out_dir, img, img_data):
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
    plt.savefig(os.path.join(fig_out_dir, img[0][:-4] + "_qual_fig" + ".png"), bbox_inches="tight")
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

    # Set up placeholders
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
            
            save_output(output, fin_out_dir, img, args)
            gt_data = load_gt_mask(args, img)
            save_qual_fig(output, gt_data, fig_out_dir, img, img_data)
            
            row = compute_metrics(output, gt_data, img, sitk_img, args)
            writer.writerow(row)

        row = ["Average Scores"]
        if args.compute_dice:
            row.append(np.mean(dice_arr))  
        if args.compute_jaccard:
            row.append(np.mean(jacc_arr))
        if args.compute_assd:  
            row.append(np.mean(assd_arr))
        writer.writerow(row)
    print("Done.")