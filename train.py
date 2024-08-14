from __future__ import print_function

import sys
sys.path.extend(['./','./Models/','./Model_Helpers/','./Data_Loader/','./Custom_Functions/'])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()
import os
import numpy as np
from tensorflow.python.keras.optimizers import Adam  
from tensorflow.python.keras import backend as K
K.set_image_data_format("channels_last")
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard
import tensorflow as tf
from Custom_Functions.custom_losses import dice_hard, weighted_binary_crossentropy_loss, dice_loss, margin_loss
from Data_Loader.load_3D_data import load_class_weights, generate_train_batches, generate_val_batches

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_loss(root, split, net, choice):
    """
    Selects and configures the loss function based on the specified choice.

    Args:
    root (str): The root directory where data is stored.
    split (int): The data split number used to load specific data.
    net (str): The name of the neural network model being used.
    choice (str): The type of loss function to use. Options include:
        - "w_bce": Weighted Binary Cross-Entropy. Uses class weights to handle class imbalance.
        - "bce": Standard Binary Cross-Entropy, a common loss function for binary classification tasks.
        - "dice": Dice Loss, which is particularly useful for segmentation tasks to maximize overlap between predicted and ground truth.
        - "w_mar": Weighted Margin Loss, which applies a margin-based loss with weights to handle class imbalance.
        - "mar": Standard Margin Loss, a margin-based loss function without class weighting.

    Returns:
    tuple: A tuple containing the configured loss function and None (as a placeholder for potential future use of loss weighting).

    Raises:
    ValueError: If an unknown loss function choice is provided.
    """
    
    if choice == "w_bce":
        pos_class_weight = load_class_weights(root=root, split=split)
        loss = weighted_binary_crossentropy_loss(pos_class_weight)
    elif choice == "bce":
        loss = "binary_crossentropy"
    elif choice == "dice":
        loss = dice_loss
    elif choice == "w_mar":
        pos_class_weight = load_class_weights(root=root, split=split)
        loss = margin_loss(margin=0.4, downweight=0.5, pos_weight=pos_class_weight)
    elif choice == "mar":  
        loss = margin_loss(margin=0.4, downweight=0.5, pos_weight=1.0)
    else:
        raise ValueError(f"Unknown loss: {choice}")
    return loss, None

def get_callbacks(args):
    monitor_name = "val_dice_hard"

    csv_logger = CSVLogger(os.path.join(args.log_dir, args.output_name + "_log" + ".csv"), separator=',')
    tb = TensorBoard(args.tf_log_dir, histogram_freq=0)
    model_checkpoint = ModelCheckpoint(
        os.path.join(args.check_dir, args.output_name + "_model_" + args.time + ".hdf5"),
        monitor=monitor_name, save_best_only=True, save_weights_only=True, verbose=1, mode='max')
    lr_reducer = ReduceLROnPlateau(monitor=monitor_name, factor=0.05, cooldown=0, patience=5, verbose=1, mode='max')
    early_stopper = EarlyStopping(monitor=monitor_name, min_delta=0, patience=25, verbose=0, mode='max')

    return [model_checkpoint, csv_logger, lr_reducer, early_stopper, tb]

def compile_model(args, net_input_shape, uncomp_model):
    opt = Adam(learning_rate=args.initial_lr, beta_1=0.99, beta_2=0.999, decay=1e-6)
    metrics = [dice_hard]
    loss, loss_weighting = get_loss(root=args.data_root_dir, split=args.split_num, net=args.net, choice=args.loss)
    
    if args.gpus <= 1:
        uncomp_model.compile(optimizer=opt, loss=loss, metrics=metrics)
        return uncomp_model
    else:
        with tf.device("/cpu:0"):
            uncomp_model.compile(optimizer=opt, loss=loss, loss_weights=loss_weighting, metrics=metrics)
        model = tf.keras.utils.multi_gpu_model(uncomp_model, gpus=args.gpus)
        model.__setattr__("callback_model", uncomp_model)
        model.compile(optimizer=opt, loss=loss, loss_weights=loss_weighting, metrics=metrics)
        return model

def plot_training(training_history, args):
    f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 10))
    subtitle = f"{args.net.upper()}-{args.activation}"
    f.suptitle(subtitle, fontsize=18)

    ax1.plot(training_history.history["dice_hard"])
    ax1.plot(training_history.history["val_dice_hard"])
    ax1.set_title("Dice Coefficient")
    ax1.set_ylabel("Dice", fontsize=12)
    ax1.legend(["Train", "Val"], loc="upper left")
    ax1.set_yticks(np.arange(0, 1.05, 0.05))
    ax1.set_xticks(np.arange(0, len(training_history.history["dice_hard"])))
    ax1.grid(linestyle="-.")
    
    ax2.plot(training_history.history["loss"])
    ax2.plot(training_history.history["val_loss"])
    ax2.set_title("Model Loss")  
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.legend(["Train", "Val"], loc="upper right")
    ax1.set_xticks(np.arange(0, len(training_history.history["loss"])))
    ax2.grid(linestyle="-.")
        
    f.savefig(os.path.join(args.output_dir, args.output_name + "_plots" + ".png"))
    plt.close()
    
def train(args, train_list, val_list, model, net_input_shape):
    model = compile_model(args=args, net_input_shape=net_input_shape, uncomp_model=model)
    if args.retrain and args.weights_path:
        print(f"\nRetrain model from {args.weights_path}")
        model.load_weights(args.weights_path)
        
    callbacks = get_callbacks(args)
    
    history = model.fit(
        generate_train_batches(
            args.data_root_dir, train_list, net_input_shape, net=args.net, batchSize=args.batch_size, numSlices=args.slices, 
            subSampAmt=args.subsamp, stride=args.stride, shuff=args.shuffle_data, aug_data=args.aug_data
        ),
        max_queue_size=40, workers=20, use_multiprocessing=False,
        steps_per_epoch=args.steps,
        validation_data=generate_val_batches(
            args.data_root_dir, val_list, net_input_shape, net=args.net, batchSize=args.batch_size,
            numSlices=args.slices, subSampAmt=0, stride=20, shuff=args.shuffle_data
        ),
        validation_steps=250,
        epochs=args.epochs, callbacks=callbacks, verbose=1,
    )
    
    plot_training(history, args)