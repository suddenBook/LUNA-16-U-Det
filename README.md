## Overview

This project focuses on lung nodule detection and classification using the [LUNA16](https://luna16.grand-challenge.org/Home/) dataset, featuring a comprehensive collection of annotated CT scans. The core of this project is the implementation of the U-Det model, a sophisticated architecture designed specifically for high accuracy in lung nodule detection. The primary objectives are to preprocess the dataset, train the U-Det model, and evaluate its effectiveness in identifying lung nodules.

## Model Architecture

For a detailed view of the U-Det model architecture used in this project, refer to the architecture diagram provided in the PDF file:

[View U-Det Model Architecture](./Tools/Visualize/Plot_CNN_Architecture/U-Det.pdf)

## Installation

To run this project, you need Python 3.10 or later. Clone the repository and install the required packages:

```bash
git clone https://github.com/suddenBook/LUNA-16-U-Det.git
cd LUNA-16-U-Det
pip install -r requirements.txt
```

## Usage

### 1. Get Dataset

The LUNA16 dataset is publicly available and can be accessed [here](https://luna16.grand-challenge.org/Download/). It consists of CT scans with annotations that mark the locations and sizes of lung nodules.

To download the LUNA16 dataset, you can also utilize the provided script [download_dataset.py](./Tools/download_dataset.py). This script will automatically download all necessary files and save them in the `Dataset` folder.

### 2. Data Preparation

After downloading the LUNA16 dataset, you will have multiple subsets (`subset0.zip` to `subset9.zip`). To streamline the preprocessing, merge all these subsets into a single directory named `subset`.

Your directory structure should look like this:

```
Dataset/
│
├── annotations.csv
├── candidates.csv
├── candidates_V2.csv
├── sampleSubmission.csv
├── subset/
│   ├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178.mhd
│   ├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178.raw
│   ├── ...
│   1776 files
│
├── seg-lungs-LUNA16/
│   ├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178.mhd
│   ├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178.raw
│   ├── ...
```

### 3. Data Preprocessing

Before running the preprocessing script, ensure you modify the data source path in the script to point to your local dataset directory.

To modify the data source path, open the `pre_process.py` file and update the `root` variable:

```python
root = os.path.normpath('/path/to/your/Dataset/')
```

After setting the correct path, execute the preprocessing script as follows:

```bash
python pre_process.py
```

This script will process the data and save the results in specified directories for masks and lung ROI images.

### 4. Model Training and Testing

The `main.py` script handles training and testing of the model. It uses command-line arguments to specify the parameters such as data directory, network type, and training options.

#### Example Usage

```bash
# Basic training example
python main.py --data_root_dir /path/to/your/Data/ --train 1 --net bifpn --epochs 30

# Training with specific GPU settings
python main.py --data_root_dir /path/to/your/Data/ --train 1 --net unet --activation relu --epochs 50 --which_gpus 0,1 --gpus 2

# Testing with pre-trained weights
python main.py --data_root_dir /path/to/your/Data/ --test 1 --weights_path ./path/to/weights.hdf5 --net unet --activation relu --split_num 1

# Using data augmentation and custom loss during training
python main.py --data_root_dir /path/to/your/Data/ --train 1 --net bifpn --epochs 30 --aug_data 1 --loss dice

# Testing with UDet_Mixed method
python main.py --data_root_dir /path/to/your/Data/ --test 1 --net udet_mixed --activation mish --split_num 1 --udet_weights /path/to/udet_weights.hdf5 --udet_small_sized_weights /path/to/udet_small_sized_weights.hdf5
```

#### Arguments

Here is a detailed description of all the arguments used in the `main.py` script:

| Argument                     | Type  | Default  | Options                                           | Description                                                                                    |
|------------------------------|-------|----------|---------------------------------------------------|------------------------------------------------------------------------------------------------|
| `--data_root_dir`            | str   | Required |                                                   | The root directory for your data.                                                              |
| `--weights_path`             | str   | ""       |                                                   | Path to the trained model weights. Set to "" if none.                                          |
| `--retrain`                  | int   | 0        | 0, 1                                              | Whether to retrain the model from scratch.                                                     |
| `--epochs`                   | int   | 20       |                                                   | Number of epochs to train the model.                                                           |
| `--steps`                    | int   | 1000     |                                                   | Number of steps per epoch.                                                                     |
| `--split_num`                | int   | 0        |                                                   | Which training split to train/test on.                                                         |
| `--net`                      | str   | "udet"   | unet, udet, bifpn, udet_small_sized, udet_mixed   | Choose your network architecture.                                                              |
| `--train`                    | int   | 0        | 0, 1                                              | Set to 1 to enable training mode.                                                              |
| `--test`                     | int   | 0        | 0, 1                                              | Set to 1 to enable testing mode.                                                               |
| `--shuffle_data`             | int   | 1        | 0, 1                                              | Whether to shuffle the training data.                                                          |
| `--aug_data`                 | int   | 1        | 0, 1                                              | Whether to use data augmentation during training.                                              |
| `--loss`                     | str   | "w_bce"  | bce, w_bce, dice, mar, w_mar                      | Which loss function to use.                                                                    |
| `--batch_size`               | int   | 2        |                                                   | Batch size for training/testing.                                                               |
| `--initial_lr`               | float | 0.0001   |                                                   | Initial learning rate.                                                                         |
| `--slices`                   | int   | 1        |                                                   | Number of slices to include for training/testing.                                              |
| `--subsamp`                  | int   | -1       |                                                   | Number of slices to skip when forming 3D samples for training.                                 |
| `--stride`                   | int   | 1        |                                                   | Number of slices to move when generating the next sample.                                      |
| `--verbose`                  | int   | 1        | 0, 1, 2                                           | Set the verbose level for training output.                                                     |
| `--save_raw`                 | int   | 1        | 0, 1                                              | Whether to save the raw output.                                                                |
| `--save_seg`                 | int   | 1        | 0, 1                                              | Whether to save the segmented output.                                                          |
| `--save_prefix`              | str   | ""       |                                                   | Prefix to append to saved CSV files.                                                           |
| `--thresh_level`             | float | 0.0      |                                                   | Threshold level for segmentation; 0.0 for Otsu's method, otherwise set value.                  |
| `--compute_dice`             | int   | 1        | 0, 1                                              | Whether to compute the Dice coefficient.                                                       |
| `--compute_jaccard`          | int   | 1        | 0, 1                                              | Whether to compute the Jaccard index.                                                          |
| `--compute_assd`             | int   | 0        | 0, 1                                              | Whether to compute the Average Symmetric Surface Distance.                                     |
| `--which_gpus`               | str   | "0"      | -2, -1, or list of GPU IDs                        | GPU settings: "-2" for CPU only, "-1" for all GPUs, or a list of GPU IDs.                      |
| `--gpus`                     | int   | -1       |                                                   | Number of GPUs to use.                                                                         |
| `--num_splits`               | int   | 4        |                                                   | Number of data splits for cross-validation.                                                    |
| `--activation`               | str   | "mish"   | relu, mish                                        | Choose the activation function for the model.                                                  |
| `--udet_weights`             | str   | ""       |                                                   | [For U-Det Mixed method only] Path to the pre-trained weights for the U-Det model.             |
| `--udet_small_sized_weights` | str   | ""       |                                                   | [For U-Det Mixed method only] Path to the pre-trained weights for the U-Det Small-Sized model. |

## References

This project utilizes concepts and code from several sources which have been instrumental in shaping the implementation of our models and preprocessing techniques.

- [U-Det: A modified U-Net architecture with bidirectional feature network for lung nodule segmentation](https://github.com/Nik-V9/U-Det)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [RadIO: A library for data science research of computed tomography imaging](https://github.com/analysiscenter/radio)
- [PlotNeuralNet: Latex code for making neural networks diagrams](https://github.com/HarisIqbal88/PlotNeuralNet)

## Citation

If you use any part of the code for any research implementation or project, please do cite the original author.

```
@misc{keetha2020udet,
    title={U-Det: A Modified U-Net architecture with bidirectional feature network for lung nodule segmentation},
    author={Nikhil Varma Keetha and Samson Anosh Babu P and Chandra Sekhara Rao Annavarapu},
    year={2020},
    eprint={2003.09293},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```