import argparse
import numpy as np
from models import resnet
from models import nnUNetv2
from models import LCOVNet
from models import SAM_Med3D
from models import UNETR
from models import Swin_UNETR

from benchmark.Kits21 import Kits21
from benchmark.LITS import LITS
from benchmark.ABD import ABD
from benchmark.LIDC import LIDC
from benchmark.COLON import COLON
from benchmark.PAN import PAN


def model_map(model_name):
    """
    Maps the input model name string to its corresponding model class or module.

    Args:
        model_name (str): Name identifier of the model.

    Returns:
        Corresponding model class/module if found, else None.
    """
    model_map = {
        'resnet' : resnet,
        'nnUNetv2' : nnUNetv2,
        'LCOVNet' : LCOVNet,
        'SAM_Med3D' : SAM_Med3D,
        'UNETR' : UNETR,
        'Swin_UNETR' : Swin_UNETR
    }
    return model_map.get(model_name)


def dataset_map(dataset_name):
    """
    Maps the dataset name to its corresponding benchmark dataset class.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        Dataset class if found, else None.
    """
    dataset_map = {
        'Kits21': Kits21,
        'LITS' : LITS,
        'ABD' : ABD,
        'LIDC' : LIDC,
        'COLON' : COLON,
        'PAN' : PAN
    }
    return dataset_map.get(dataset_name)


def evaluate(logits, label):
    """
    Computes the Dice coefficient between the predicted logits and ground-truth labels.

    Args:
        logits (np.ndarray): Predicted binary mask.
        label (np.ndarray): Ground-truth binary mask.

    Returns:
        float: Dice score between prediction and ground truth.
    """
    logits = logits.astype(np.float32)
    label = label.astype(np.float32)
    inter = np.dot(logits.flatten(), label.flatten())
    union = np.sum(logits) + np.sum(label)
    dice = (2 * inter + 1e-5) / (union + 1e-5)
    return dice


def min_max_normalize(arr):
    """
    Performs min-max normalization on the input array to scale it into [0, 1].

    Args:
        arr (np.ndarray): Input array to normalize.

    Returns:
        np.ndarray: Normalized array.
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val == 0:
        return np.zeros_like(arr)  # Prevent division by zero
    return (arr - min_val) / (max_val - min_val)


def parse_opts():
    """
    Parses command-line arguments for training, validation, and testing procedures.
    This function sets up all configurable parameters for model, dataset, learning rate,
    training epochs, input dimensions, GPU usage, and other runtime settings.

    Returns:
        argparse.Namespace: Parsed argument object containing all configurations.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--val_every', default=1, type=int,
                        help='Validation frequency in number of epochs.')

    parser.add_argument('--data_root', default='./data', type=str,
                        help='Root directory path containing dataset.')

    parser.add_argument('--img_list', default='./data/ULS23/train.txt', type=str,
                        help='Path to the training image list.')

    parser.add_argument('--val_list', default='./data/ULS23/calibration.txt', type=str,
                        help='Path to the calibration/validation image list.')

    parser.add_argument('--n_seg_classes', default=2, type=int,
                        help='Number of segmentation output classes.')

    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Initial learning rate for optimization.')

    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of parallel data loading workers.')

    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for training.')

    parser.add_argument('--phase', default='train', type=str,
                        help='Execution phase: train or test.')

    parser.add_argument('--save_intervals', default=10, type=int,
                        help='Interval (in epochs) to save model checkpoints.')

    parser.add_argument('--n_epochs', default=200, type=int,
                        help='Total number of training epochs.')

    parser.add_argument('--input_D', default=256, type=int,
                        help='Input volume depth (z-axis size).')

    parser.add_argument('--input_H', default=256, type=int,
                        help='Input volume height (y-axis size).')

    parser.add_argument('--input_W', default=128, type=int,
                        help='Input volume width (x-axis size).')

    parser.add_argument('--resume_path', default='', type=str,
                        help='Path to resume training from a saved checkpoint.')

    parser.add_argument('--pretrain_path', default='pretrain/resnet_50.pth', type=str,
                        help='Path to a pretrained model.')

    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA if set.')

    parser.set_defaults(no_cuda=False)

    parser.add_argument('--pin_memory', dest='pin_memory', action='store_true',
                        help='Pin memory in DataLoader for faster GPU transfer.')

    parser.set_defaults(pin_memory=True)

    parser.add_argument('--gpu_id', nargs='+', type=int,
                        help='List of GPU device IDs to use.')

    parser.add_argument('--model', default='Med3D', type=str,
                        help='Model architecture to use.')

    parser.add_argument('--dataset', default='COLON', type=str,
                        help='Dataset identifier.')

    parser.add_argument('--manual_seed', default=1, type=int,
                        help='Seed for reproducibility.')

    parser.add_argument('--ci_test', action='store_true',
                        help='Enable CI (continuous integration) testing mode.')

    args = parser.parse_args()

    args.save_folder = "./trails/{}/{}_{}".format(args.model, args.model, args.model_depth)

    return args
