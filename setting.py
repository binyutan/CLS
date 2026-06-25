import argparse

from models import med3d
from models import nnunet
from models import sam_med3d
from models import unetr
from models import swin_unetr

from benchmark.kits21 import Kits21
from benchmark.lits import LITS
from benchmark.abd import ABD
from benchmark.lidc import LIDC
from benchmark.colon import COLON
from benchmark.pan import PAN


def model_map(model_name):
    """
    Maps the input model name string to its corresponding model module.

    Args:
        model_name (str): Name identifier of the model.

    Returns:
        Corresponding model module if found, else None.
    """
    _model_map = {
        'Med3D': med3d,
        'nnUNet': nnunet,
        'UNETR': unetr,
        'SwinUNETR': swin_unetr,
        'SAMMed3D': sam_med3d,
    }
    return _model_map.get(model_name)


def dataset_map(dataset_name):
    """
    Maps the dataset name to its corresponding benchmark dataset class.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        Dataset class if found, else None.
    """
    _dataset_map = {
        'Kits21': Kits21,
        'LITS': LITS,
        'ABD': ABD,
        'LIDC': LIDC,
        'COLON': COLON,
        'PAN': PAN,
    }
    return _dataset_map.get(dataset_name)


def parse_opts():
    """
    Parses command-line arguments for training, calibration, and testing.

    Returns:
        argparse.Namespace: Parsed argument object containing all configurations.
    """
    parser = argparse.ArgumentParser()

    # ---- Data ----
    parser.add_argument('--data_root', default='./data', type=str,
                        help='Root directory path containing dataset.')
    parser.add_argument('--img_list', default='./data/ULS23/train.txt', type=str,
                        help='Path to the training image list.')
    parser.add_argument('--val_list', default='./data/ULS23/calibration.txt', type=str,
                        help='Path to the calibration/validation image list.')
    parser.add_argument('--dataset', default='COLON', type=str,
                        choices=['Kits21', 'LITS', 'ABD', 'LIDC', 'COLON', 'PAN'],
                        help='Dataset identifier.')

    # ---- Model ----
    parser.add_argument('--model', default='Med3D', type=str,
                        choices=['Med3D', 'nnUNet', 'UNETR', 'SwinUNETR', 'SAMMed3D'],
                        help='Model architecture to use.')
    parser.add_argument('--n_seg_classes', default=2, type=int,
                        help='Number of segmentation output classes.')
    parser.add_argument('--pretrain_path', default='', type=str,
                        help='Path to a pretrained model.')
    parser.add_argument('--resume_path', default='', type=str,
                        help='Path to resume training from a saved checkpoint.')

    # ---- Training ----
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for training.')
    parser.add_argument('--n_epochs', default=200, type=int,
                        help='Total number of training epochs.')
    parser.add_argument('--val_every', default=1, type=int,
                        help='Validation frequency in number of epochs.')
    parser.add_argument('--save_intervals', default=10, type=int,
                        help='Interval (in epochs) to save model checkpoints.')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of parallel data loading workers.')

    # ---- Input dimensions ----
    parser.add_argument('--input_D', default=256, type=int,
                        help='Input volume depth (z-axis).')
    parser.add_argument('--input_H', default=256, type=int,
                        help='Input volume height (y-axis).')
    parser.add_argument('--input_W', default=128, type=int,
                        help='Input volume width (x-axis).')

    # ---- CLS: Conformal Lesion Segmentation ----
    parser.add_argument('--risk_level', default=0.2, type=float,
                        help='FNR tolerance epsilon.')
    parser.add_argument('--alpha', default=0.2, type=float,
                        help='Violation probability alpha.')
    parser.add_argument('--n_splits', default=100, type=int,
                        help='Number of random calibration-test splits.')
    parser.add_argument('--cal_ratio', default=0.5, type=float,
                        help='Proportion of data used for calibration.')

    # ---- GPU ----
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA if set.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--pin_memory', dest='pin_memory', action='store_true',
                        help='Pin memory in DataLoader.')
    parser.set_defaults(pin_memory=True)
    parser.add_argument('--gpu_id', nargs='+', type=int,
                        help='List of GPU device IDs to use.')

    # ---- Misc ----
    parser.add_argument('--phase', default='train', type=str,
                        help='Execution phase: train or test.')
    parser.add_argument('--manual_seed', default=1, type=int,
                        help='Seed for reproducibility.')
    parser.add_argument('--ci_test', action='store_true',
                        help='Enable CI testing mode.')

    args = parser.parse_args()

    args.save_folder = "./trials/{}/{}".format(args.model, args.dataset)

    return args
