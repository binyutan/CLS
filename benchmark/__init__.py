import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import ndimage
import nibabel as nib

from utils.file_process import load_lines


class BaseSegDataset(Dataset):
    """
    Base 3D lesion segmentation dataset for the ULS23 challenge.

    Each line in img_list is formatted as:
        relative/path/to/image.nii.gz  relative/path/to/label.nii.gz

    Returns (image, label) where:
        image: Tensor of shape (1, D, H, W)
        label: Tensor of shape (D, H, W)
    """

    def __init__(self, data_root, img_list, args):
        lines = load_lines(img_list)
        self.img_paths = [os.path.join(data_root, l.split()[0]) for l in lines]
        self.label_paths = [os.path.join(data_root, l.split()[1]) for l in lines]
        self.input_D = args.input_D
        self.input_H = args.input_H
        self.input_W = args.input_W

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = nib.load(self.img_paths[idx]).get_fdata().astype(np.float32)
        label = nib.load(self.label_paths[idx]).get_fdata().astype(np.float32)

        if img.ndim == 4:
            img = img[..., 0]
        if label.ndim == 4:
            label = label[..., 0]

        img = self._preprocess(img)
        label = self._preprocess_label(label)

        scale = [self.input_D / img.shape[0],
                 self.input_H / img.shape[1],
                 self.input_W / img.shape[2]]
        img = ndimage.zoom(img, scale, order=1)
        label = ndimage.zoom(label, scale, order=0)

        img = torch.from_numpy(img).unsqueeze(0).float()
        label = torch.from_numpy(label).float()
        return img, label

    def _preprocess(self, img):
        img = np.clip(img, -1000, 1000)
        min_val, max_val = img.min(), img.max()
        if max_val - min_val > 0:
            img = (img - min_val) / (max_val - min_val)
        return img

    def _preprocess_label(self, label):
        return (label > 0).astype(np.float32)
