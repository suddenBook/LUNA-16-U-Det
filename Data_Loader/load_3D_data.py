import os
from os.path import join
from glob import glob
import csv
from sklearn.model_selection import KFold
import numpy as np
from numpy.random import shuffle
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow.keras.preprocessing.image as preprocess
import matplotlib.pyplot as plt
import threading

import sys
sys.path.extend(['./','../','../Models/','../Custom_Functions/','../Model_Helpers/'])

from Custom_Functions.custom_data_aug import elastic_transform, salt_pepper_noise

debug = 0

class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

def load_data(data_root_dir, split_num):
    with open(join(data_root_dir, "split_lists", f"train_split_{split_num}.csv"), "r") as f:
        reader = csv.reader(f)
        training_list = [row for row in reader if row]

    with open(join(data_root_dir, "split_lists", f"test_split_{split_num}.csv"), "r") as f:
        reader = csv.reader(f)
        testing_list = [row for row in reader if row]

    split_index = int(0.8 * len(training_list))
    new_training_list = training_list[:split_index]
    validation_list = training_list[split_index:]

    return new_training_list, validation_list, testing_list

def compute_class_weights(root, train_data_list):
    pos = neg = 0.0
    for img_name in tqdm(train_data_list):
        img = np.load(join(root, "masks", "masks_" + img_name[0])).T
        for slic in img:
            if np.any(slic):
                p = np.count_nonzero(slic)
                pos += p
                neg += slic.size - p
    return neg / pos

def load_class_weights(root, split):
    class_weight_filename = join(root, "split_lists", f"train_split_{split}_class_weights.npy")
    try:
        return np.load(class_weight_filename)
    except FileNotFoundError:
        print(f"Class weight file {class_weight_filename} not found.\nComputing class weights now. This may take some time.")
        train_data_list, _, _ = load_data(root, str(split))
        value = compute_class_weights(root, train_data_list)
        np.save(class_weight_filename, value)
        print("Finished computing class weights. This value has been saved for this training split.")
        return value

def split_data(root_path, num_splits):
    """
    Splits the dataset into training and testing sets based on the number of splits specified.

    Args:
    root_path (str): The root directory where the mask files are located.
    num_splits (int): The number of splits to create. If num_splits is 1, the same index is used for both training and testing.

    This function first collects all mask files from the specified directory, then divides them into the specified number of splits.
    Each split's indices are saved into separate CSV files for training and testing datasets.
    """
    mask_list = []
    for ext in ("*.mhd", "*.hdr", "*.nii", "*.npy"):
        mask_list.extend(sorted(glob(join(root_path, "masks", ext))))

    assert mask_list, f"Unable to find any files in {join(root_path, 'masks')}"

    outdir = join(root_path, "split_lists")
    os.makedirs(outdir, exist_ok=True)

    if num_splits == 1:
        train_index = test_index = [0]
        for name, index in (("train", train_index), ("test", test_index)):
            with open(join(outdir, f"{name}_split_0.csv"), "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                for i in index:
                    writer.writerow([os.path.basename(mask_list[i]).replace("masks_", "")])
    else:
        kf = KFold(n_splits=num_splits)
        for n, (train_index, test_index) in enumerate(kf.split(mask_list)):
            for name, index in (("train", train_index), ("test", test_index)):
                with open(join(outdir, f"{name}_split_{n}.csv"), "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    for i in index:
                        writer.writerow([os.path.basename(mask_list[i]).replace("masks_", "")])

def convert_data_to_numpy(root_path, img_name, no_masks=False, overwrite=False):
    """
    Converts image and mask data from file formats to numpy arrays and saves them as compressed NPZ files.

    Args:
    root_path (str): The root directory where the image and mask files are located.
    img_name (str): The name of the image file to be converted.
    no_masks (bool, optional): If True, does not load or convert mask data. Default is False.
    overwrite (bool, optional): If True, overwrites existing NPZ files. Default is False.

    Returns:
    tuple: A tuple containing numpy arrays for the image and mask, or just the image if no_masks is True.

    This function checks if the NPZ files already exist and either loads them or creates them by loading the original image and mask files,
    applying necessary transformations, and then saving them as compressed NPZ files.
    """
    fname = img_name[:-4]
    numpy_path = join(root_path, "np_files")
    os.makedirs(join(numpy_path, "images"), exist_ok=True)
    if not no_masks:
        os.makedirs(join(numpy_path, "masks"), exist_ok=True)

    if not overwrite and os.path.isfile(join(numpy_path, f"{fname}.npz")):
        with np.load(join(numpy_path, f"{fname}.npz")) as data:
            return data["img"], data["mask"]

    try:
        img = np.load(join(root_path, "imgs", "images_" + img_name))
        assert len(img.shape) == 3, f"Expected 3D image, got {img.ndim}D"

        if not no_masks:
            mask = np.load(join(root_path, "masks", "masks_" + img_name))
            assert len(mask.shape) == 3, f"Expected 3D mask, got {mask.ndim}D"

        if no_masks:
            np.savez_compressed(join(numpy_path, "images", f"images_{fname}.npz"), img=img)
            return img
        else:
            np.savez_compressed(join(numpy_path, "images", f"images_{fname}.npz"), img=img)
            np.savez_compressed(join(numpy_path, "masks", f"masks_{fname}.npz"), mask=mask)
            return img, mask
    except Exception as e:
        print(f"\n{'-'*100}\nUnable to load img or masks for {fname}\n{e}\nSkipping file\n{'-'*100}\n")
        return np.zeros(1), np.zeros(1)

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def augment_data(imgs, masks, img_aug=True, mask_aug=True):
    for i in range(len(imgs)):
        if img_aug:
            if np.random.rand() < 0.1:
                imgs[i] = preprocess.random_rotation(
                    imgs[i], 45, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant', cval=0)
            if np.random.rand() < 0.2:
                imgs[i] = elastic_transform(imgs[i], alpha=1000, sigma=80, alpha_affine=50)
            if np.random.rand() < 0.1:
                imgs[i] = preprocess.random_shift(
                    imgs[i], 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant', cval=0)
            if np.random.rand() < 0.1:
                imgs[i] = preprocess.random_shear(
                    imgs[i], 16, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant', cval=0)
            if np.random.rand() < 0.1:
                imgs[i] = preprocess.random_zoom(
                    imgs[i], (0.75,0.75), row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant', cval=0)
            if np.random.rand() < 0.1:
                imgs[i] = flip_axis(imgs[i], axis=1)
            if np.random.rand() < 0.1:
                imgs[i] = flip_axis(imgs[i], axis=0)
            if np.random.rand() < 0.1:
                imgs[i] = salt_pepper_noise(imgs[i])

        if mask_aug:
            if np.random.rand() < 0.1:
                masks[i] = preprocess.random_rotation(
                    masks[i], 45, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant', cval=0)
            if np.random.rand() < 0.2:
                masks[i] = elastic_transform(masks[i], alpha=1000, sigma=80, alpha_affine=50)
            if np.random.rand() < 0.1:
                masks[i] = preprocess.random_shift(
                    masks[i], 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant', cval=0)
            if np.random.rand() < 0.1:
                masks[i] = preprocess.random_shear(
                    masks[i], 16, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant', cval=0)
            if np.random.rand() < 0.1:
                masks[i] = preprocess.random_zoom(
                    masks[i], (0.75,0.75), row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant', cval=0)
            if np.random.rand() < 0.1:
                masks[i] = flip_axis(masks[i], axis=1)
            if np.random.rand() < 0.1:
                masks[i] = flip_axis(masks[i], axis=0)

        masks[masks > 0.5] = 1
        masks[masks <= 0.5] = 0

    return imgs, masks

@threadsafe_generator
def generate_train_batches(
    root_path,
    train_list,
    net_input_shape,
    net,
    batchSize,
    numSlices=1,
    subSampAmt=-1,
    stride=1,
    shuff=True,
    aug_data=True,
):
    while True:
        if shuff:
            shuffle(train_list)

        targ_shape = np.concatenate(((batchSize,), net_input_shape))
        img_batch = np.zeros(targ_shape, dtype=np.float32)
        mask_batch = np.zeros_like(img_batch)

        count = 0
        
        for scan_name in train_list:
            try:
                img_path = os.path.normpath(os.path.join(root_path, "imgs", "images_" + scan_name[0]))
                mask_path = os.path.normpath(os.path.join(root_path, "masks", "masks_" + scan_name[0]))
                
                img_npz_path = os.path.normpath(os.path.join(root_path, "np_files", "images", f"images_{scan_name[0][:-4]}.npz"))
                mask_npz_path = os.path.normpath(os.path.join(root_path, "np_files", "masks", f"masks_{scan_name[0][:-4]}.npz"))
                
                if os.path.exists(img_npz_path) and os.path.exists(mask_npz_path):
                    img_data = np.load(img_path)
                    img = img_data.T
                    mask_data = np.load(mask_path)
                    mask = mask_data.T
                    print(f"\nTrain: Pre-made numpy array exists as {scan_name[0][:-4]}.")
                    print(f"Shape of train_img: {img.shape}")
                    assert len(img.shape) == 3, f"Expected 3D array, got {len(img.shape)}D array"
                else:
                    raise FileNotFoundError(f"NPZ files not found for {scan_name[0][:-4]}")
            except Exception as e:
                print("\n", e)
                print(f"Train: Pre-made numpy array not found for {scan_name[0][:-4]}.\nCreating now...")
                img, mask = convert_data_to_numpy(root_path, scan_name[0])
                if np.array_equal(img, np.zeros(1)):
                    continue
                print("Finished making npz file.")

            if numSlices == 1:
                subSampAmt = 0
            elif subSampAmt == -1 and numSlices > 1:
                np.random.seed(None)
                subSampAmt = int(np.random.rand() * (img.shape[2] * 0.05))

            indicies = np.arange(0, img.shape[2] - numSlices * (subSampAmt + 1) + 1, stride)
            if shuff:
                shuffle(indicies)

            for j in indicies:
                if not np.any(mask[:, :, j : j + numSlices * (subSampAmt + 1) : subSampAmt + 1]):
                    continue
                img_batch[count] = img[:, :, j : j + numSlices * (subSampAmt + 1) : subSampAmt + 1]
                mask_batch[count] = mask[:, :, j : j + numSlices * (subSampAmt + 1) : subSampAmt + 1]
                count += 1
                if count == batchSize:
                    if aug_data:
                        img_batch, mask_batch = augment_data(img_batch, mask_batch)
                    if debug:
                        plt.imshow(np.squeeze(img_batch[0]), cmap='gray')
                        plt.imshow(np.squeeze(mask_batch[0]), alpha=0.15)
                        plt.savefig(join(root_path, "logs", "ex_train.png"), format="png", bbox_inches="tight")
                        plt.close()
                    yield ([img_batch, mask_batch], [mask_batch, mask_batch * img_batch]) if 'caps' in net else (img_batch, mask_batch)

                    count = 0
                    img_batch = np.zeros(targ_shape, dtype=np.float32)
                    mask_batch = np.zeros_like(img_batch)

        if count > 0:
            if aug_data:
                img_batch[:count], mask_batch[:count] = augment_data(img_batch[:count], mask_batch[:count])
            yield ([img_batch[:count], mask_batch[:count]], [mask_batch[:count], mask_batch[:count] * img_batch[:count]]) if 'caps' in net else (img_batch[:count], mask_batch[:count])

@threadsafe_generator
def generate_val_batches(
    root_path,
    val_list,
    net_input_shape,
    net,
    batchSize,
    numSlices=1,
    subSampAmt=-1,
    stride=1,
    shuff=True
):
    while True:
        if shuff:
            shuffle(val_list)

        targ_shape = np.concatenate(((batchSize,), net_input_shape))
        img_batch = np.zeros(targ_shape, dtype=np.float32)
        mask_batch = np.zeros_like(img_batch)

        count = 0
        for scan_name in val_list:
            try:
                img_path = os.path.normpath(os.path.join(root_path, "imgs", "images_" + scan_name[0]))
                mask_path = os.path.normpath(os.path.join(root_path, "masks", "masks_" + scan_name[0]))
                
                img_npz_path = os.path.normpath(os.path.join(root_path, "np_files", "images", f"images_{scan_name[0][:-4]}.npz"))
                mask_npz_path = os.path.normpath(os.path.join(root_path, "np_files", "masks", f"masks_{scan_name[0][:-4]}.npz"))
                
                if os.path.exists(img_npz_path) and os.path.exists(mask_npz_path):
                    img_data = np.load(img_path)
                    img = img_data.T
                    mask_data = np.load(mask_path)
                    mask = mask_data.T
                    print(f"\nValidate: Pre-made numpy array exists as {scan_name[0][:-4]}.")
                    print(f"Shape of train_img: {img.shape}")
                    assert len(img.shape) == 3, f"Expected 3D array, got {len(img.shape)}D array"
                else:
                    raise FileNotFoundError(f"NPZ files not found for {scan_name[0][:-4]}")
            except Exception as e:
                print("\n", e)
                print(f"Validate: Pre-made numpy array not found for {scan_name[0][:-4]}.\nCreating now...")
                img, mask = convert_data_to_numpy(root_path, scan_name[0])
                if np.array_equal(img, np.zeros(1)):
                    continue
                print("Finished making npz file.")

            if numSlices == 1:
                subSampAmt = 0
            elif subSampAmt == -1 and numSlices > 1:
                np.random.seed(None)
                subSampAmt = int(np.random.rand() * (img.shape[2] * 0.05))

            indicies = np.arange(0, img.shape[2] - numSlices * (subSampAmt + 1) + 1, stride)
            if shuff:
                shuffle(indicies)

            for j in indicies:
                if not np.any(mask[:, :, j : j + numSlices * (subSampAmt + 1) : subSampAmt + 1]):
                    continue
                img_batch[count] = img[:, :, j : j + numSlices * (subSampAmt + 1) : subSampAmt + 1]
                mask_batch[count] = mask[:, :, j : j + numSlices * (subSampAmt + 1) : subSampAmt + 1]
                count += 1
                if count == batchSize:
                    yield ([img_batch, mask_batch], [mask_batch, mask_batch * img_batch]) if 'caps' in net else (img_batch, mask_batch)
                    count = 0
                    img_batch = np.zeros(targ_shape, dtype=np.float32)
                    mask_batch = np.zeros_like(img_batch)
                    
        if count > 0:
            yield ([img_batch[:count], mask_batch[:count]], [mask_batch[:count], mask_batch[:count] * img_batch[:count]]) if 'caps' in net else (img_batch[:count], mask_batch[:count])

@threadsafe_generator
def generate_test_batches(
    root_path, 
    test_list,
    net_input_shape,
    batchSize,
    numSlices=1,  
    subSampAmt=0,
    stride=1
):
    targ_shape = np.concatenate(((batchSize,), net_input_shape))
    img_batch = np.zeros(targ_shape, dtype=np.float32)
    
    for scan_name in test_list:
        try:
            img_path = os.path.normpath(os.path.join(root_path, "imgs", "images_" + scan_name[0]))
            img_npz_path = os.path.normpath(os.path.join(root_path, "np_files", "images", f"images_{scan_name[0][:-4]}.npz"))
            
            if os.path.exists(img_npz_path):
                img_data = np.load(img_path)
                img = img_data.T
                print(f"\nValidate: Pre-made numpy array exists as {scan_name[0][:-4]}.")
                print(f"Shape of train_img: {img.shape}")
                assert len(img.shape) == 3, f"Expected 3D array, got {len(img.shape)}D array"
            else:
                raise FileNotFoundError(f"NPZ files not found for {scan_name[0][:-4]}")
        except Exception as e:
            print("\n", e)
            print(f"Test: Pre-made numpy array not found for {scan_name[0][:-4]}.\nCreating now...")
            img = convert_data_to_numpy(root_path, scan_name[0], no_masks=True)
            if np.array_equal(img, np.zeros(1)):
                continue
            print("Finished making npz file.")
        
        if numSlices == 1:
            subSampAmt = 0
        elif subSampAmt == -1 and numSlices > 1:
            np.random.seed(None)
            subSampAmt = int(np.random.rand() * (img.shape[2] * 0.05))
            
        indicies = np.arange(0, img.shape[2] - numSlices * (subSampAmt + 1) + 1, stride)
        count = 0
        for j in indicies:
            img_batch[count] = img[:, :, j : j + numSlices * (subSampAmt + 1) : subSampAmt + 1]
            count += 1
            if count == batchSize:
                yield img_batch
                count = 0
                img_batch = np.zeros(targ_shape, dtype=np.float32)
        if count > 0:
            yield img_batch[:count]