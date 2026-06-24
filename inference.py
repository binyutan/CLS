from setting import model_map
from setting import dataset_map
from setting import parse_opts

import json
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy import ndimage
import nibabel as nib
import sys
import os
from utils.file_process import load_lines
import matplotlib.pyplot as plt

def seg_eval(pred, label, clss):
    """
    Compute Dice scores between the predicted mask and ground truth.

    Args:
        pred (ndarray): Predicted binary segmentation mask.
        label (ndarray): Ground truth mask.
        clss (list): List of class labels to evaluate.

    Returns:
        dices (ndarray): Dice score for each class.
    """
    Ncls = len(clss)
    dices = np.zeros(Ncls)
    [depth, height, width] = pred.shape
    for idx, cls in enumerate(clss):
        # Create binary masks for the current class
        pred_cls = np.zeros([depth, height, width])
        pred_cls[np.where(pred == cls)] = 1

        label_cls = np.zeros([depth, height, width])
        label_cls[np.where(label == cls)] = 1

        # Compute intersection and union
        s = pred_cls + label_cls
        inter = len(np.where(s >= 2)[0])
        conv = len(np.where(s >= 1)[0]) + inter
        try:
            dice = 2.0 * inter / conv
        except:
            print("conv is zero when computing Dice")
            dice = -1

        dices[idx] = dice

    return dices

def save_res_img(img, label_flag, idx):
    """
    Save three orthogonal slices of a 3D volume as visualization.

    Args:
        img (ndarray): 3D image or prediction.
        label_flag (bool): Whether it is label or prediction.
        idx (int): Index for naming the saved file.
    """
    z = img.shape[0] // 2
    y = img.shape[1] // 2
    x = img.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img[z, :, :], cmap='jet')
    axes[0].set_title('Axial (Z)')
    axes[1].imshow(img[:, y, :], cmap='jet')
    axes[1].set_title('Coronal (Y)')
    axes[2].imshow(img[:, :, x], cmap='jet')
    axes[2].set_title('Sagittal (X)')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()

    flag = 'label' if label_flag else 'predict'
    save_path = f"res_img/{flag}_slice_{idx}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def compute_loss(pred, label):
    """
    Compute false negative ratio (FNR)-based loss.

    Args:
        pred (ndarray): Binarized prediction.
        label (ndarray): Ground truth mask.

    Returns:
        loss (float): Loss value (1 - recall).
    """
    assert pred.shape == label.shape, "Shapes must match."

    intersection = np.sum((pred == 1) & (label == 1))
    label_positive = np.sum(label == 1)

    if label_positive == 0:
        return 0.0

    ratio = intersection / label_positive
    return 1 - ratio

def find_threshold_for_loss(mask, label, target_loss, tol=1e-4, max_iter=200):
    """
    Perform binary search to find the minimum threshold 
    such that the loss is less than or equal to a target value.

    Args:
        mask (ndarray): Probability map.
        label (ndarray): Ground truth mask.
        target_loss (float): Desired upper bound on loss.
        tol (float): Numerical tolerance.
        max_iter (int): Maximum iterations.

    Returns:
        threshold (float): Calibrated threshold.
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
    # Load configuration
    sets = parse_opts()
    sets.target_type = "normal"
    sets.phase = 'test'

    # Load model from checkpoint
    checkpoint = torch.load(sets.resume_path)
    net, _, _, _ = model_map(sets.model).generate_model(sets)
    net.load_state_dict(checkpoint['state_dict'])

    # Create DataLoader for test dataset
    testing_data = dataset_map(sets.dataset)(sets.data_root, sets.img_list, sets)
    data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    # Load file names
    img_names = [info.split(" ")[0] for info in load_lines(sets.img_list)]
    label_names = [info.split(" ")[1] for info in load_lines(sets.img_list)]

    # Generate probability maps using the model
    masks = model_map(sets.model).test(data_loader, net, img_names, sets)

    Nimg = len(img_names)
    dices = np.zeros([Nimg, sets.n_seg_classes])

    # Post-process each image
    for idx in range(Nimg):
        base_name = os.path.basename(img_names[idx]).split('.')[0]
        save_path = os.path.join(f"res/probability_matrix/{sets.model}/{sets.dataset}", base_name + ".npy")

        # Save probability map
        np.save(save_path, masks[idx].cpu().numpy())

        # Load ground truth label
        label = nib.load(os.path.join(sets.data_root, label_names[idx])).get_fdata()
        if label.ndim == 4 and label.shape[-1] == 1:
            label = np.squeeze(label, axis=-1)

        # Binarize prediction at fixed threshold (e.g., 0.5)
        masks[idx] = (masks[idx].cpu().numpy() > 1 - 0.5).astype(int)
        dices[idx, :] = seg_eval(masks[idx], label, range(sets.n_seg_classes))

    # Print mean Dice score for each class (excluding background)
    for idx in range(1, sets.n_seg_classes):
        mean_dice_per_task = np.mean(dices[:, idx])
        print('Mean Dice for class-{} is {:.4f}'.format(idx, mean_dice_per_task))
