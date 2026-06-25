"""
nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.
Isensee et al., Nature Methods, 2021.

Replace the placeholder model with the actual nnU-Net v2 architecture.
Installation: pip install nnunetv2
Official repository: https://github.com/MIC-DKFZ/nnUNet

Expected input:  (B, 1, D, H, W) float32 CT volume
Expected output: (B, n_seg_classes, D, H, W) logits
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.amp import autocast
import torch.nn.functional as F
from scipy import ndimage
import nibabel as nib


class nnUNetSegNet(nn.Module):
    """Placeholder — replace with actual nnU-Net architecture."""
    def __init__(self, n_seg_classes=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1), nn.InstanceNorm3d(32), nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 64, 3, padding=1), nn.InstanceNorm3d(64), nn.LeakyReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(64, 32, 3, padding=1), nn.InstanceNorm3d(32), nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, n_seg_classes, 1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def generate_model(args):
    model = nnUNetSegNet(n_seg_classes=args.n_seg_classes)
    if not args.no_cuda:
        model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.99, nesterov=True)
    scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.n_epochs)
    loss_func = nn.CrossEntropyLoss()
    return model, optimizer, scheduler, loss_func


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    total_loss = 0.0
    for batch_data in loader:
        imgs, labels = batch_data
        if not args.no_cuda:
            imgs, labels = imgs.cuda(), labels.cuda()
        optimizer.zero_grad()
        with autocast():
            logits = model(imgs)
            loss = loss_func(logits, labels.long())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def val_epoch(model, loader, epoch, loss_func, args):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_data in loader:
            imgs, labels = batch_data
            if not args.no_cuda:
                imgs, labels = imgs.cuda(), labels.cuda()
            logits = model(imgs)
            loss = loss_func(logits, labels.long())
            total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def test(data_loader, net, img_names, args):
    """Run inference and return per-image lesion probability maps (numpy)."""
    net.eval()
    masks = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            volume = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data
            if not args.no_cuda:
                volume = volume.cuda()
            probs = F.softmax(net(volume), dim=1)
            _, _, md, mh, mw = probs.shape

            data = nib.load(os.path.join(args.data_root, img_names[batch_id])).get_fdata()
            if data.ndim == 4 and data.shape[-1] == 1:
                data = data.squeeze(-1)
            d, h, w = data.shape

            mask = probs[0].cpu().numpy()
            scale = [1, d / md, h / mh, w / mw]
            mask = ndimage.zoom(mask, scale, order=1)
            masks.append(mask[1])
    return masks
