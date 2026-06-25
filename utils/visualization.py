import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_res_img(img, label_flag, idx, save_dir='res_img'):
    os.makedirs(save_dir, exist_ok=True)
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
    save_path = os.path.join(save_dir, f'{flag}_slice_{idx}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_ecr_vs_alpha(ecr_dict, alpha_values, epsilon, dataset, save_dir='res/figures'):
    """Fig.3: ECR vs risk level alpha for each model."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))

    markers = ['o', 's', '^', 'D', 'v']
    for i, (model_name, ecr_values) in enumerate(ecr_dict.items()):
        marker = markers[i % len(markers)]
        ax.plot(alpha_values, ecr_values, marker=marker, label=model_name, linewidth=2)

    lower_bound = [1 - a for a in alpha_values]
    ax.plot(alpha_values, lower_bound, 'k--', linewidth=1.5, label='lower bound')

    ax.set_xlabel(r'Risk Level ($\alpha$)', fontsize=13)
    ax.set_ylabel('ECR', fontsize=13)
    ax.set_title(f'{dataset} ($\\varepsilon$ = {epsilon})', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'ecr_vs_alpha_{dataset}.pdf'), dpi=300)
    plt.close()


def plot_fnr_fpr_scatter(fnr_list, fpr_list, model, dataset, calibrated, save_dir='res/figures'):
    """Fig.4: Per-sample V-FNR vs V-FPR scatter plot."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(fpr_list, fnr_list, alpha=0.5, s=15, edgecolors='none')

    mean_fnr = np.mean(fnr_list)
    mean_fpr = np.mean(fpr_list)

    label = 'CLS' if calibrated else 'Heuristic'
    ax.set_title(f'{model} ({label})\nMean FNR: {mean_fnr:.3f}\nMean FPR: {mean_fpr:.3f}', fontsize=11)
    ax.set_xlabel('FPR', fontsize=12)
    ax.set_ylabel('FNR', fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    suffix = 'calibrated' if calibrated else 'heuristic'
    plt.savefig(os.path.join(save_dir, f'fnr_fpr_{model}_{dataset}_{suffix}.pdf'), dpi=300)
    plt.close()


def plot_pc_bar(pc_dict, risk_levels, alpha_values, dataset, save_dir='res/figures'):
    """Fig.5a: Prediction Compactness across models and risk levels."""
    os.makedirs(save_dir, exist_ok=True)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    models = list(pc_dict.keys())
    x_pos = np.arange(len(alpha_values))
    width = 0.15

    for m_idx, model_name in enumerate(models):
        for r_idx, rl in enumerate(risk_levels):
            key = f'{rl}'
            if key in pc_dict[model_name]:
                values = pc_dict[model_name][key]
                xs = x_pos + m_idx * width
                ys = [r_idx] * len(xs)
                ax.bar3d(xs, ys, 0, width * 0.8, 0.8, values, alpha=0.7, label=model_name if r_idx == 0 else '')

    ax.set_xlabel(r'Risk Level ($\alpha$)', fontsize=11)
    ax.set_ylabel(r'$\varepsilon$', fontsize=11)
    ax.set_zlabel('PC (x1e-2)', fontsize=11)
    ax.set_title(f'Prediction Compactness on {dataset}', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'pc_bar_{dataset}.pdf'), dpi=300)
    plt.close()


def plot_ecr_vs_cal_ratio(ecr_dict, cal_ratios, epsilon, alpha, dataset, save_dir='res/figures'):
    """Fig.5b: ECR vs calibration set proportion."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))

    markers = ['o', 's', '^', 'D', 'v']
    for i, (model_name, ecr_values) in enumerate(ecr_dict.items()):
        marker = markers[i % len(markers)]
        ax.plot(cal_ratios, ecr_values, marker=marker, label=model_name, linewidth=2)

    ax.axhline(y=1 - alpha, color='k', linestyle='--', linewidth=1.5, label='lower bound')

    ax.set_xlabel('Calibration Set Proportion', fontsize=13)
    ax.set_ylabel('ECR', fontsize=13)
    ax.set_title(f'{dataset} ($\\varepsilon$ = {epsilon}, $\\alpha$ = {alpha})', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'ecr_vs_cal_ratio_{dataset}.pdf'), dpi=300)
    plt.close()
