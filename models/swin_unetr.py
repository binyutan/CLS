"""
Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images.
Hatamizadeh et al., MICCAI BrainLes Workshop, 2021.

Replace the placeholder model with the actual Swin-UNETR from MONAI:
    from monai.networks.nets import SwinUNETR as MonaiSwinUNETR
Official documentation: https://docs.monai.io/en/stable/networks.html

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


class SwinUNETRSegNet(nn.Module):
    """Placeholder — replace with monai.networks.nets.SwinUNETR."""
    def __init__(self, n_seg_classes=2, img_size=(256, 256, 128)):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 48, 3, padding=1), nn.BatchNorm3d(48), nn.GELU(),
            nn.Conv3d(48, 96, 3, padding=1), nn.BatchNorm3d(96), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(96, 48, 3, padding=1), nn.BatchNorm3d(48), nn.GELU(),
            nn.Conv3d(48, n_seg_classes, 1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def generate_model(args):
    model = SwinUNETRSegNet(
        n_seg_classes=args.n_seg_classes,
        img_size=(args.input_D, args.input_H, args.input_W),
    )
    if not args.no_cuda:
        model = model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=5e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
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
