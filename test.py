from setting import parse_opts

import os
import json
import numpy as np
import nibabel as nib

from utils.file_process import load_lines, ensure_dir
from utils.metrics import (
    min_max_normalize,
    apply_threshold,
    compute_fnr_loss,
    compute_fpr_loss,
    compute_ecr,
    compute_pfnr,
    compute_prediction_compactness,
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

    # Load calibration JSON
    cal_json_path = os.path.join("res/json", sets.model, f"{sets.model}_{sets.dataset}.json")
    with open(cal_json_path, "r") as f:
        log_dict = json.load(f)

    all_ecr = []
    all_pfnr = []
    all_vfnr = []
    all_vfpr = []
    all_pc = []

    for split_id in range(n_splits):
        print(f"\n=== Split {split_id} ===")
        split_file = f"data/ULS23/split/{sets.dataset}/shuffled/val_{split_id + 1}.txt"

        img_names = [info.split(" ")[0] for info in load_lines(split_file)]
        label_names = [info.split(" ")[1] for info in load_lines(split_file)]
        Nimg = len(label_names)

        # Get calibrated threshold
        split_key = str(split_id)
        try:
            t_hat = log_dict["Sampling"][split_key]["t_hat"][str(risk_level)][str(alpha)]
        except KeyError:
            print(f"  ERROR: No calibration data for epsilon={risk_level}, alpha={alpha} "
                  f"in split {split_id}. Run calibration.py with matching parameters first.")
            continue
        print(f"  t_hat: {t_hat:.6f}")

        # Determine test indices
        n_cal = int(Nimg * sets.cal_ratio)
        test_indices = list(range(n_cal, Nimg))

        fnr_list = []
        fpr_list = []
        pc_list = []

        for idx in test_indices:
            # Load ground truth
            label = nib.load(os.path.join(sets.data_root, label_names[idx])).get_fdata()
            if label.ndim == 4 and label.shape[-1] == 1:
                label = np.squeeze(label, axis=-1)

            base_name = os.path.splitext(os.path.basename(img_names[idx]))[0]

            # Load probability mask and apply calibrated threshold
            prob_path = os.path.join(
                f"res/probability_matrix/{sets.model}/{sets.dataset}", base_name + ".npy")
            mask = np.load(prob_path)
            mask = min_max_normalize(mask)

            pred = apply_threshold(mask, t_hat)

            # Compute metrics (Eq.3, Eq.4)
            fnr = compute_fnr_loss(pred, label)
            fpr = compute_fpr_loss(pred, label)
            pc = compute_prediction_compactness(pred)

            fnr_list.append(fnr)
            fpr_list.append(fpr)
            pc_list.append(pc)

            print(f"  test[{idx}] FNR: {fnr:.4f}  FPR: {fpr:.6f}  PC: {pc:.6f}")

        # Compute aggregate metrics for this split
        ecr = compute_ecr(fnr_list, risk_level)
        pfnr = compute_pfnr(fnr_list)
        mean_vfnr = np.mean(fnr_list)
        mean_vfpr = np.mean(fpr_list)
        mean_pc = np.mean(pc_list)

        all_ecr.append(ecr)
        all_pfnr.append(pfnr)
        all_vfnr.append(mean_vfnr)
        all_vfpr.append(mean_vfpr)
        all_pc.append(mean_pc)

        print(f"  ECR: {ecr:.4f}  P-FNR: {pfnr:.4f}  "
              f"V-FNR: {mean_vfnr:.4f}  V-FPR: {mean_vfpr:.6f}  PC: {mean_pc:.6f}")

        # Store per-split results back to JSON
        log_dict["Sampling"][split_key]["loss_values"][str(risk_level)][str(alpha)] = fnr_list
        log_dict["Sampling"][split_key]["fpr_values"] = {str(risk_level): {str(alpha): fpr_list}}
        log_dict["Sampling"][split_key]["pc_values"] = {str(risk_level): {str(alpha): pc_list}}
        log_dict["Sampling"][split_key]["ecr"] = {str(risk_level): {str(alpha): ecr}}

    # Save updated results
    out_json_path = os.path.join(
        "res/json", sets.model, f"{alpha}_{sets.model}_{sets.dataset}.json")
    ensure_dir(os.path.dirname(out_json_path))
    with open(out_json_path, "w") as f:
        json.dump(log_dict, f, indent=2)

    # ---- Summary Table ----
    print("\n" + "=" * 70)
    print(f"Summary: {sets.model} on {sets.dataset}")
    print(f"  epsilon = {risk_level}, alpha = {alpha}, splits = {n_splits}")
    print("-" * 70)
    print(f"  {'Metric':<20} {'Mean':>10} {'Std':>10}")
    print("-" * 70)
    print(f"  {'ECR':<20} {np.mean(all_ecr):>10.4f} {np.std(all_ecr):>10.4f}")
    print(f"  {'V-FNR':<20} {np.mean(all_vfnr):>10.4f} {np.std(all_vfnr):>10.4f}")
    print(f"  {'P-FNR':<20} {np.mean(all_pfnr):>10.4f} {np.std(all_pfnr):>10.4f}")
    print(f"  {'V-FPR':<20} {np.mean(all_vfpr):>10.6f} {np.std(all_vfpr):>10.6f}")
    print(f"  {'PC':<20} {np.mean(all_pc):>10.6f} {np.std(all_pc):>10.6f}")
    print("=" * 70)
    print(f"\nResults saved to {out_json_path}")
