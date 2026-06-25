import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def min_max_normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


# ---------------------------------------------------------------------------
# Eq(1): Binarization  Mask_i(t) = {(d,h,w) : f(x_i)_{d,h,w} >= 1 - t}
# ---------------------------------------------------------------------------

def apply_threshold(prob_map, t):
    return (prob_map >= 1.0 - t).astype(np.int32)


# ---------------------------------------------------------------------------
# Eq(3): Voxel-level FNR-specific loss
# L_i^FNR(t) = 1 - |Mask_i(t) ∩ y_i*| / |y_i*|
# ---------------------------------------------------------------------------

def compute_fnr_loss(pred, label):
    label_positive = np.sum(label == 1)
    if label_positive == 0:
        return 0.0
    tp = np.sum((pred == 1) & (label == 1))
    return 1.0 - tp / label_positive


def compute_fnr_loss_gpu(pred, label):
    label_positive = torch.sum(label == 1)
    if label_positive == 0:
        return torch.tensor(0.0, device=pred.device)
    tp = torch.sum((pred == 1) & (label == 1))
    return 1.0 - tp.float() / label_positive.float()


# ---------------------------------------------------------------------------
# Eq(4): Voxel-level FPR-specific loss
# L_i^FPR(t) = |Mask_i(t) ∩ (1 - y_i*)| / |1 - y_i*|
# ---------------------------------------------------------------------------

def compute_fpr_loss(pred, label):
    bg = np.sum(label == 0)
    if bg == 0:
        return 0.0
    fp = np.sum((pred == 1) & (label == 0))
    return fp / bg


def compute_fpr_loss_gpu(pred, label):
    bg = torch.sum(label == 0)
    if bg == 0:
        return torch.tensor(0.0, device=pred.device)
    fp = torch.sum((pred == 1) & (label == 0))
    return fp.float() / bg.float()


# ---------------------------------------------------------------------------
# Eq(5): Nonconformity score  t_i = inf{t : L_i^FNR(t) <= epsilon}
# Binary search for the smallest t such that FNR(t) <= epsilon.
# ---------------------------------------------------------------------------

def find_threshold_for_loss(mask, label, target_loss, tol=1e-4, max_iter=200):
    low, high = 0.0, 1.0
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        binarized = (mask > 1.0 - mid).astype(int)
        loss = compute_fnr_loss(binarized, label)
        if abs(loss - target_loss) < tol:
            return mid
        elif loss > target_loss:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def find_threshold_for_loss_gpu(mask, label, target_loss, tol=1e-4, max_iter=200):
    low, high = 0.0, 1.0
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        binarized = (mask > 1.0 - mid).int()
        loss = compute_fnr_loss_gpu(binarized, label)
        if abs(loss.item() - target_loss) < tol:
            return mid
        elif loss.item() > target_loss:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


# ---------------------------------------------------------------------------
# Eq(6): Conformal quantile
# t_hat = sorted_scores[ceil((1-alpha)(1+n)) - 1]
# ---------------------------------------------------------------------------

def compute_conformal_quantile(scores, alpha):
    n = len(scores)
    sorted_scores = sorted(scores)
    k = int(np.ceil((1.0 - alpha) * (1.0 + n)))
    k = min(k, n)
    return sorted_scores[k - 1]


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_ecr(loss_list, epsilon):
    return np.mean([l <= epsilon for l in loss_list])


def compute_pfnr(loss_list):
    return np.mean([l == 1.0 for l in loss_list])


def compute_prediction_compactness(pred):
    return np.sum(pred == 1) / pred.size


def seg_eval(pred, label, clss):
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
        except ZeroDivisionError:
            dice = 0.0
        dices[idx] = dice
    return dices


def evaluate(logits, label):
    logits = logits.astype(np.float32)
    label = label.astype(np.float32)
    inter = np.dot(logits.flatten(), label.flatten())
    union = np.sum(logits) + np.sum(label)
    return (2 * inter + 1e-5) / (union + 1e-5)
