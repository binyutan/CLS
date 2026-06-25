from setting import parse_opts

import os
import json
import torch
import numpy as np
import nibabel as nib

from utils.file_process import load_lines, ensure_dir
from utils.metrics import (
    min_max_normalize,
    compute_fnr_loss,
    find_threshold_for_loss_gpu,
    compute_conformal_quantile,
)


if __name__ == '__main__':
    sets = parse_opts()
    sets.target_type = "normal"
    sets.phase = 'test'

    risk_level = sets.risk_level
    alpha = sets.alpha
    n_splits = sets.n_splits
    print(f"risk_level (epsilon): {risk_level}")
    print(f"alpha: {alpha}")
    print(f"n_splits: {n_splits}")

    # ---- Phase 1 & 2: Loop over random calibration-test splits ----
    log_path = os.path.join("res/json", sets.model, f"{sets.model}_{sets.dataset}.json")
    ensure_dir(os.path.dirname(log_path))
    log_dict = {"Sampling": {}}

    for split_id in range(n_splits):
        print(f"\n=== Split {split_id} ===")
        split_file = f"data/ULS23/split/{sets.dataset}/shuffled/val_{split_id + 1}.txt"

        img_names = [info.split(" ")[0] for info in load_lines(split_file)]
        label_names = [info.split(" ")[1] for info in load_lines(split_file)]
        Nimg = len(label_names)

        # Determine calibration / test indices based on cal_ratio
        n_cal = int(Nimg * sets.cal_ratio)
        cal_indices = list(range(n_cal))
        test_indices = list(range(n_cal, Nimg))

        # ---- Phase 1: Compute nonconformity scores t_i for calibration set (Eq.5) ----
        t_values = []
        for idx in cal_indices:
            label_data = nib.load(os.path.join(sets.data_root, label_names[idx])).get_fdata()
            if label_data.ndim == 4 and label_data.shape[-1] == 1:
                label_data = np.squeeze(label_data, axis=-1)

            base_name = os.path.splitext(os.path.basename(img_names[idx]))[0]
            mask = np.load(os.path.join(
                f"res/probability_matrix/{sets.model}/{sets.dataset}", base_name + ".npy"))
            mask = min_max_normalize(mask)

            mask_gpu = torch.from_numpy(mask).float().cuda()
            label_gpu = torch.from_numpy(label_data).int().cuda()

            t_i = find_threshold_for_loss_gpu(mask_gpu, label_gpu, risk_level)
            t_values.append(t_i)
            print(f"  cal[{idx}] t_i: {t_i:.6f}")

        # ---- Phase 2: Compute conformal quantile t_hat (Eq.6) ----
        t_hat = compute_conformal_quantile(t_values, alpha)
        print(f"  t_hat (epsilon={risk_level}, alpha={alpha}): {t_hat:.6f}")

        # Store results per split
        log_dict["Sampling"][str(split_id)] = {
            "t_values": {str(risk_level): t_values},
            "t_hat": {str(risk_level): {str(alpha): t_hat}},
            "loss_values": {str(risk_level): {str(alpha): []}},
            "cal_indices": cal_indices,
            "test_indices": test_indices,
        }

    # Save calibration results
    with open(log_path, "w") as f:
        json.dump(log_dict, f, indent=2)
    print(f"\nCalibration results saved to {log_path}")
