from __future__ import print_function

import sys
sys.path.extend(['./','./Models/','./Model_Helpers/','./Data_Loader/','./Custom_Functions/'])

import os
import argparse
import SimpleITK as sitk
from time import gmtime, strftime
import numpy as np
from load_3D_data import load_data, split_data
from model_helper import create_model
from train import train
from test import test

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and test a 3D medical image segmentation model.",
        epilog="Example usage: python main.py --data_root_dir /path/to/data --train 1 --net udet --epochs 30"
    )
    parser.add_argument("--data_root_dir", type=str, required=True, help="The root directory for your data.")
    parser.add_argument("--weights_path", type=str, default="", help='/path/to/trained_model.hdf5 from root. Set to "" for none.')
    parser.add_argument("--retrain", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--split_num", type=int, default=0, help="Which training split to train/test on.")
    parser.add_argument("--net", type=str.lower, default="udet", choices=["unet", "udet", "bifpn"], help="Choose your network.")
    parser.add_argument("--train", type=int, default=0, choices=[0,1], help="Set to 1 to enable training.")
    parser.add_argument("--test", type=int, default=0, choices=[0,1], help="Set to 1 to enable testing.")
    parser.add_argument("--shuffle_data", type=int, default=1, choices=[0,1], help="Whether to shuffle the training data.")
    parser.add_argument("--aug_data", type=int, default=1, choices=[0,1], help="Whether to use data augmentation during training.")
    parser.add_argument("--loss", type=str.lower, default="w_bce", choices=["bce", "w_bce", "dice", "mar", "w_mar"], help="Which loss to use.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training/testing.")
    parser.add_argument("--initial_lr", type=float, default=0.0001, help="Initial learning rate.")
    parser.add_argument("--slices", type=int, default=1, help="Number of slices to include for training/testing.")
    parser.add_argument("--subsamp", type=int, default=-1, help="Number of slices to skip when forming 3D samples for training.")
    parser.add_argument("--stride", type=int, default=1, help="Number of slices to move when generating the next sample.")
    parser.add_argument("--verbose", type=int, default=1, choices=[0,1,2], help="Set the verbose value for training.")
    parser.add_argument("--save_raw", type=int, default=1, choices=[0,1], help="Whether to save the raw output.")
    parser.add_argument("--save_seg", type=int, default=1, choices=[0,1], help="Whether to save the segmented output.")
    parser.add_argument("--save_prefix", type=str, default="", help="Prefix to append to saved CSV.")
    parser.add_argument("--thresh_level", type=float, default=0.0, help="Enter 0.0 for otsu thresholding, else set value.")
    parser.add_argument("--compute_dice", type=int, default=1, help="0 or 1")
    parser.add_argument("--compute_jaccard", type=int, default=1, help="0 or 1")  
    parser.add_argument("--compute_assd", type=int, default=0, help="0 or 1")
    parser.add_argument("--which_gpus", type=str, default="0", help='Enter "-2" for CPU only, "-1" for all GPUs available, or a comma separated list of GPU id numbers.')
    parser.add_argument("--gpus", type=int, default=-1, help="Number of GPUs to use.")
    parser.add_argument("--num_splits", type=int, default=4)
    parser.add_argument("--activation", type=str.lower, default="mish", choices=["relu", "mish"], help="Choose the activation function for the model.")
    return parser.parse_args()

def load_train_val_test(args):
    try:
        train_list, val_list, test_list = load_data(args.data_root_dir, args.split_num)
    except:
        split_data(args.data_root_dir, num_splits=args.num_splits)
        train_list, val_list, test_list = load_data(args.data_root_dir, args.split_num)
    return train_list, val_list, test_list

def make_output_dirs(args):
    args.time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    
    args.output_name = (
        f"{args.net}_{args.activation}_"
        f"split-{args.split_num}_batch-{args.batch_size}_shuff-{args.shuffle_data}_aug-{args.aug_data}_"
        f"loss-{args.loss}_slic-{args.slices}_sub-{args.subsamp}_strid-{args.stride}_lr-{args.initial_lr}"
    )

    args.check_dir = os.path.join(args.data_root_dir, "saved_models", args.net, args.activation)
    os.makedirs(args.check_dir, exist_ok=True)
    
    args.log_dir = os.path.join(args.data_root_dir, "logs", args.net, args.activation)
    os.makedirs(args.log_dir, exist_ok=True)

    args.tf_log_dir = os.path.join(args.log_dir, "tf_logs")
    os.makedirs(args.tf_log_dir, exist_ok=True)

    args.output_dir = os.path.join(args.data_root_dir, "plots", args.net, args.activation)
    os.makedirs(args.output_dir, exist_ok=True)

def main(args):
    assert args.train or args.test, "Cannot have train and test both set to 0"
    
    train_list, val_list, test_list = load_train_val_test(args)
    
    image = np.load(os.path.join(args.data_root_dir, "imgs", "images_" + train_list[0][0])).T 
    image = image[np.newaxis, :, :]
    net_input_shape = (image.shape[1], image.shape[2], args.slices)

    model_list = create_model(args=args, input_shape=net_input_shape)
    model = model_list[0]
    model.summary(positions=[0.38, 0.65, 0.75, 1.0])

    make_output_dirs(args)
    
    if args.which_gpus == "-2":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.which_gpus == "-1":
        assert args.gpus != -1, (
            "With all GPUs option, you must specify the number of GPUs with --gpus"
        )
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.which_gpus)
        args.gpus = len(args.which_gpus.split(","))

    if args.gpus > 1:
        assert args.batch_size >= args.gpus, (
            "Batch size must be >= number of GPUs used."
        )
        
    if args.train:
        train(args, train_list, val_list, model, net_input_shape)
        
    if args.test:
        test(args, test_list, model_list, net_input_shape)

if __name__ == "__main__":
    args = parse_args()
    main(args)