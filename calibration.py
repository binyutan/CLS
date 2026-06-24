from setting import model_map
from setting import dataset_map
from setting import parse_opts
from setting import min_max_normalize

import json
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy import ndimage
import nibabel as nib
import os
from utils.file_process import load_lines
import matplotlib.pyplot as plt


def seg_eval(pred, label, clss):
    """
    Compute Dice scores between predicted mask and ground truth.

    Args:
        pred (ndarray): Binarized predicted mask.
        label (ndarray): Ground truth label.
        clss (list): List of class labels (e.g., [0, 1] for binary segmentation).

    Returns:
        ndarray: Dice score for each class.
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
            print("conv is zero during Dice computation.")
            dice = -1
        dices[idx] = dice

    return dices


def save_res_img(img, label_flag, idx):
    """
    Save orthogonal slices of 3D image for visualization.

    Args:
        img (ndarray): 3D input image or mask.
        label_flag (bool): Whether the image is label (True) or prediction (False).
        idx (int): Index used for naming output file.
    """
    z, y, x = img.shape[0] // 2, img.shape[1] // 2, img.shape[2] // 2
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
    Compute false negative–based loss (1 - recall).

    Args:
        pred (ndarray): Binarized predicted mask.
        label (ndarray): Ground truth.

    Returns:
        float: Loss value.
    """
    assert pred.shape == label.shape
    intersection = np.sum((pred == 1) & (label == 1))
    label_positive = np.sum(label == 1)
    if label_positive == 0:
        return 0.0
    return 1 - intersection / label_positive


def test(data_loader, model, img_names, sets):
    """
    Run model inference and return class probability masks.

    Args:
        data_loader (DataLoader): Input volume loader.
        model (torch.nn.Module): Segmentation model.
        img_names (list): Image file paths.
        sets (argparse.Namespace): Parsed arguments.

    Returns:
        list of ndarray: Probability masks for each image.
    """
    masks = []
    model.eval()
    for batch_id, batch_data in enumerate(data_loader):
        volume = batch_data.cuda() if not sets.no_cuda else batch_data
        with torch.no_grad():
            probs = F.softmax(model(volume), dim=1)

        [_, _, mask_d, mask_h, mask_w] = probs.shape
        data = nib.load(os.path.join(sets.data_root, img_names[batch_id])).get_fdata()
        if data.ndim == 4 and data.shape[-1] == 1:
            data = np.squeeze(data, axis=-1)
        [depth, height, width] = data.shape

        mask = probs[0]
        scale = [1, depth / mask_d, height / mask_h, width / mask_w]
        mask = ndimage.zoom(mask.cpu().numpy(), scale, order=1)

        masks.append(mask[1])  # Extract lesion class only

    return masks


def find_threshold_for_loss(mask, label, target_loss, tol=1e-4, max_iter=200):
    """
    Binary search to find threshold satisfying FNR-based loss constraint.

    Args:
        mask (ndarray): Probability map.
        label (ndarray): Ground truth.
        target_loss (float): Maximum allowed loss.
        tol (float): Numerical tolerance.
        max_iter (int): Maximum iterations.

    Returns:
        float: Optimal threshold.
    """
    low, high = 0.0, 1.0
    for _ in range(max_iter):
        mid = (low + high) / 2
        loss = compute_loss((mask > 1 - mid).astype(int), label)
        if abs(loss - target_loss) < tol:
            return mid
        elif loss > target_loss:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def compute_loss_gpu(pred, label):
    """
    GPU-based FNR loss for threshold optimization.

    Args:
        pred (Tensor): Binary prediction mask.
        label (Tensor): Ground truth tensor.

    Returns:
        Tensor: Loss value.
    """
    assert pred.shape == label.shape
    intersection = torch.sum((pred == 1) & (label == 1))
    label_positive = torch.sum(label == 1)
    if label_positive == 0:
        return torch.tensor(0.0, device=pred.device)
    return 1 - intersection.float() / label_positive.float()


def find_threshold_for_loss_gpu(mask, label, target_loss, tol=1e-4, max_iter=200):
    """
    GPU-accelerated binary search to find decision threshold.

    Args:
        mask (Tensor): Probability map.
        label (Tensor): Ground truth label.
        target_loss (float): Target loss level.
        tol (float): Tolerance for convergence.
        max_iter (int): Max number of search iterations.

    Returns:
        float: Calibrated threshold.
    """
    low, high = 0.0, 1.0
    for _ in range(max_iter):
        mid = (low + high) / 2
        loss = compute_loss_gpu((mask > 1 - mid).int(), label)
        if abs(loss.item() - target_loss) < tol:
            return mid
        elif loss.item() > target_loss:
            low = mid
        else:
            high = mid
    return (low + high) / 2


if __name__ == '__main__':
    # Setup and parse arguments
    sets = parse_opts()
    sets.target_type = "normal"
    sets.phase = 'test'

    risk_level = 0.4
    print("risk_level:", risk_level)

    # Load image and label file paths
    img_names = [info.split(" ")[0] for info in load_lines(sets.img_list)]
    label_names = [info.split(" ")[1] for info in load_lines(sets.img_list)]

    Nimg = len(label_names)
    dices = np.zeros([Nimg, sets.n_seg_classes])

    # Load or create calibration log
    log_path = f"res/json/{sets.model}/train_{sets.model}_{sets.dataset}.json"
    log_dict = {"Samples": {}} if not os.path.exists(log_path) else json.load(open(log_path))

    # Iterate over each sample
    for idx in range(Nimg):
        label_data = nib.load(os.path.join(sets.data_root, label_names[idx])).get_fdata()
        if label_data.ndim == 4 and label_data.shape[-1] == 1:
            label_data = np.squeeze(label_data, axis=-1)

        base_name = os.path.splitext(os.path.basename(img_names[idx]))[0]
        mask = np.load(os.path.join(f"res/probability_matrix/{sets.model}/{sets.dataset}", base_name + ".npy"))
        mask = min_max_normalize(mask)

        # GPU acceleration for threshold search
        mask_gpu = torch.from_numpy(mask).float().to(0)
        label_gpu = torch.from_numpy(label_data).int().to(0)

        t = find_threshold_for_loss_gpu(mask_gpu, label_gpu, risk_level)
        print(idx, "t:", t, end='  ')

        # Apply threshold and evaluate
        mask_bin = (mask > 1 - t).astype(int)
        dices[idx, :] = seg_eval(mask_bin, label_data, range(sets.n_seg_classes))
        print("loss:", compute_loss(mask_bin, label_data))

        log_dict["Samples"][base_name] = {
            "t_values": {risk_level: t}
        }

    # Save updated threshold values
    with open(log_path, "w") as f:
        json.dump(log_dict, f)

    # Report mean Dice for each class
    for idx in range(sets.n_seg_classes):
        mean_dice = np.mean(dices[:, idx])
        print('Mean Dice for class-{} is {:.4f}'.format(idx, mean_dice))
