from setting import model_map
from setting import dataset_map
from setting import parse_opts
from setting import min_max_normalize

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy import ndimage
import nibabel as nib
import os
from utils.file_process import load_lines
import json
import matplotlib.pyplot as plt
import time


def seg_eval(pred, label, clss):
    """
    Compute Dice similarity coefficient for each class.

    Args:
        pred (ndarray): Binarized predicted segmentation mask.
        label (ndarray): Ground truth label volume.
        clss (list): List of class indices (e.g., [0, 1] for binary).

    Returns:
        ndarray: Dice scores for all specified classes.
    """
    Ncls = len(clss)
    dices = np.zeros(Ncls)
    [depth, height, width] = pred.shape

    for idx, cls in enumerate(clss):
        pred_cls = np.zeros([depth, height, width])
        pred_cls[np.where(pred == cls)] = 1

        label_cls = np.zeros([depth, height, width])
        label_cls[np.where(label == cls)] = 1

        s = pred_cls + label_cls
        inter = len(np.where(s >= 2)[0])
        conv = len(np.where(s >= 1)[0]) + inter
        try:
            dice = 2.0 * inter / conv
        except:
            print("conv is zero in dice = 2.0 * inter / conv")
            dice = -1
        dices[idx] = dice

    return dices


def save_res_img(img, label_flag, idx):
    """
    Save orthogonal slices (axial, coronal, sagittal) of 3D volume.

    Args:
        img (ndarray): 3D image or mask.
        label_flag (bool): Whether the input is label or prediction.
        idx (int): Index used for saving the image.
    """
    z = img.shape[0] // 2
    y = img.shape[1] // 2
    x = img.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img[z, :, :], cmap='jet'); axes[0].set_title('Axial (Z)')
    axes[1].imshow(img[:, y, :], cmap='jet'); axes[1].set_title('Coronal (Y)')
    axes[2].imshow(img[:, :, x], cmap='jet'); axes[2].set_title('Sagittal (X)')
    for ax in axes: ax.axis('off')
    plt.tight_layout()

    flag = 'label' if label_flag else 'predict'
    save_path = f"res_img/{flag}_slice_{idx}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def compute_loss(pred, label):
    """
    Compute FNR-style loss (1 - recall).

    Args:
        pred (ndarray): Binary prediction mask.
        label (ndarray): Ground truth label.

    Returns:
        float: Loss value.
    """
    assert pred.shape == label.shape
    intersection = np.sum((pred == 1) & (label == 1))
    label_positive = np.sum(label == 1)

    if label_positive == 0:
        return 0.0
    return 1 - intersection / label_positive


def compute_loss_gpu(pred, label):
    """
    Compute GPU-compatible version of 1 - recall loss.

    Args:
        pred (Tensor): Binary prediction mask.
        label (Tensor): Ground truth label.

    Returns:
        Tensor: Loss value on GPU.
    """
    assert pred.shape == label.shape
    intersection = torch.sum((pred == 1) & (label == 1))
    label_positive = torch.sum(label == 1)

    if label_positive == 0:
        return torch.tensor(0.0, device=pred.device)

    return 1 - intersection.float() / label_positive.float()


def find_threshold_for_loss(mask, label, target_loss, tol=1e-4, max_iter=200):
    """
    Perform binary search to determine optimal decision threshold under loss constraint.

    Args:
        mask (ndarray): Probability mask.
        label (ndarray): Ground truth mask.
        target_loss (float): Desired loss value.
        tol (float): Acceptable error range.
        max_iter (int): Maximum search iterations.

    Returns:
        float: Calibrated threshold.
    """
    low, high = 0.0, 1.0
    for _ in range(max_iter):
        mid = (low + high) / 2
        binarized_mask = (mask > 1 - mid).astype(int)
        loss = compute_loss(binarized_mask, label)

        if abs(loss - target_loss) < tol:
            return mid
        elif loss > target_loss:
            low = mid
        else:
            high = mid
    return (low + high) / 2


if __name__ == '__main__':
    # Configuration
    sets = parse_opts()
    sets.target_type = "normal"
    sets.phase = 'test'

    risk_level = 0.4
    alpha = 0.9
    print("risk_level:", risk_level)
    print("alpha:", alpha)

    # Load JSON logs
    with open(f"res/json/{sets.model}/{sets.model}_{sets.dataset}.json", "r") as f:
        log_dict = json.load(f)

    # Loop over 100-fold cross-validation sets
    for i in range(100):
        print(i)
        img_list = f'data/ULS23/split/{sets.dataset}/shuffled/val_{i+1}.txt'

        img_names = [info.split(" ")[0] for info in load_lines(img_list)]
        label_names = [info.split(" ")[1] for info in load_lines(img_list)]
        Nimg = len(label_names)
        dices = np.zeros([Nimg, sets.n_seg_classes])
        loss_list = []

        print(Nimg)
        for idx in range(Nimg):
            # Load ground truth
            label = nib.load(os.path.join(sets.data_root, label_names[idx])).get_fdata()
            if label.ndim == 4 and label.shape[-1] == 1:
                label = np.squeeze(label, axis=-1)

            base_name = os.path.splitext(os.path.basename(img_names[idx]))[0]

            # Load probability mask
            path = os.path.join(f"res/probability_matrix/{sets.model}/{sets.dataset}", base_name + ".npy")
            mask = np.load(path)
            mask = min_max_normalize(mask)

            # Retrieve calibrated threshold
            t_hat = log_dict["Sampling"][f"{i}"]["t_hat"][f"{risk_level}"][f"{alpha}"]
            print(idx, "t_hat:", t_hat, end='  ')

            # Threshold and compute loss
            mask = (mask > 1 - t_hat).astype(int)
            mask_gpu = torch.from_numpy(mask).float().to(0)
            label_gpu = torch.from_numpy(label).int().to(0)

            loss = compute_loss_gpu(mask_gpu, label_gpu)
            loss_list.append(loss.item())
            print("loss:", loss.item())

        # Update JSON log with per-image loss values
        log_dict["Sampling"][str(i)]["loss_values"][str(risk_level)][str(alpha)] = loss_list

    # Save updated logs to file
    with open(f"res/json/{sets.model}/{alpha}_{sets.model}_{sets.dataset}.json", "w") as f:
        json.dump(log_dict, f)

    
