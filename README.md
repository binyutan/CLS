# Conformal 3D Lesion Segmentation with Balanced Risk Control

**[Medical Image Computing and Computer Assisted Intervention (MICCAI) 2026]**

[[Paper]](https://arxiv.org/) | [[Code]](https://github.com/binyutan/CLS)

> Binyu Tan, Zhiyuan Wang, Jinhao Duan, Kaidi Xu, Heng Tao Shen, Fumin Shen, Xiaoshuang Shi

## Abstract

Medical image segmentation models commonly binarize voxel-wise scores using a fixed threshold, such as 0.5. However, this heuristic provides no finite-sample guarantee against clinically consequential under-segmentation, particularly in 3D lesion segmentation, where lesions are sparse relative to the background. We propose **Conformal Lesion Segmentation (CLS)**, a model-agnostic post-processing framework for controlling voxel-level false-negative risk. Given a user-specified FNR tolerance $\varepsilon$ and violation probability $\alpha$, CLS uses a held-out calibration set to compute, for each case, the smallest mask-expansion parameter for which the voxel-level FNR does not exceed $\varepsilon$. A finite-sample-corrected conformal quantile of these critical values is then used to determine a global binarization threshold. Under exchangeability, CLS guarantees that a new test case satisfies the prescribed FNR tolerance with probability at least $1 - \alpha$. We evaluate CLS on six 3D-LS benchmarks across five backbone models, demonstrating its superior statistical validity and predictive performance, and providing potential guidance for deploying risk-aware segmentation models in real-world clinical applications.

## Method Overview

CLS follows a four-stage pipeline:

```
                    +-----------+         +-------------+
   Training Data -> |  Stage 1  | ------> |   Stage 2   |
                    |   Train   |         |  Inference  |
                    +-----------+         +------+------+
                     Segmentation                |
                     Model f(.)           Probability Maps
                                                 |
                                                 v
                    +-----------+         +------+------+
                    |  Stage 4  | <------ |   Stage 3   |
                    |   Test    |         | Calibration |
                    +-----------+         +-------------+
                     Apply t_hat           Eq(5): t_i per sample
                     Compute ECR,          Eq(6): t_hat = quantile
                     FNR, FPR, PC
```

**Key equations:**
- **Eq(1)** Mask binarization: $\text{Mask}_i(t) = \\{(d,h,w) : f(x_i)_{d,h,w} \geq 1-t\\}$
- **Eq(3)** FNR loss: $L_i^{FNR}(t) = 1 - \frac{|\text{Mask}_i(t) \cap y_i^*|}{|y_i^*|}$
- **Eq(5)** Nonconformity score: $t_i = \inf\\{t : L_i^{FNR}(t) \leq \varepsilon\\}$
- **Eq(6)** Conformal quantile: $\hat{t} = t_{\lceil(1-\alpha)(1+n)\rceil}$

## Key Contributions

- **CLS framework**: Extends split conformal prediction to binary segmentation, bounding the probability that per-case voxel-level FNR exceeds a user-specified tolerance.
- **Novel nonconformity score**: Derived from FNR-constrained critical thresholds, enabling statistically rigorous segmentation with inherently low FPR.
- **Prediction Compactness (PC)**: An interpretable auxiliary metric quantifying spatial precision under formal risk constraints.

## Project Structure

```
code/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── setting.py                 # Configuration and argument parsing
├── train.py                   # Stage 1: Model training
├── inference.py               # Stage 2: Probability map generation
├── calibration.py             # Stage 3: Conformal calibration (Eq.5 + Eq.6)
├── test.py                    # Stage 4: Risk-controlled testing with ECR/PC
├── models/
│   ├── __init__.py
│   ├── med3d.py               # Med3D (ResNet-based) [Chen et al., 2019]
│   ├── nnunet.py              # nnU-Net [Isensee et al., 2021]
│   ├── unetr.py               # UNETR [Hatamizadeh et al., 2022]
│   ├── swin_unetr.py          # Swin-UNETR [Hatamizadeh et al., 2021]
│   └── sam_med3d.py           # SAM-Med3D [Wang et al., 2023]
├── benchmark/
│   ├── __init__.py            # BaseSegDataset class
│   ├── kits21.py              # KiTS21 kidney tumor dataset
│   ├── lits.py                # LiTS liver tumor dataset
│   ├── abd.py                 # NIH-LN ABD lymph node dataset
│   ├── lidc.py                # LIDC-IDRI lung nodule dataset
│   ├── colon.py               # MDSC-Colon dataset
│   └── pan.py                 # MDSC-Pancreas dataset
└── utils/
    ├── __init__.py
    ├── file_process.py        # File I/O utilities
    ├── metrics.py             # All evaluation metrics (Eq.1-6, ECR, PC, P-FNR)
    └── visualization.py       # Figure generation (ECR plots, FNR-FPR scatter, PC)
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA-capable GPU (recommended)

```bash
pip install -r requirements.txt
```

## Data Preparation

We use six datasets from the [ULS23 Segmentation Challenge](https://uls23.grand-challenge.org/):

| Dataset | Anatomy | Reference |
|---------|---------|-----------|
| KiTS21 | Kidney tumor | Heller et al., 2023 |
| LiTS | Liver tumor | Bilic et al., 2023 |
| NIH-LN ABD | Abdominal lymph node | Roth et al., 2014 |
| LIDC-IDRI | Lung nodule | Armato et al., 2011 |
| MDSC-Colon | Colon cancer | Antonelli et al., 2022 |
| MDSC-Pancreas | Pancreas tumor | Antonelli et al., 2022 |

Organize the data as follows:

```
data/
└── ULS23/
    ├── train.txt                          # Training image-label pairs
    ├── calibration.txt                    # Calibration image-label pairs
    └── split/
        └── {dataset}/
            └── shuffled/
                ├── val_1.txt              # Split 1
                ├── val_2.txt              # Split 2
                └── ...                    # Up to val_100.txt
```

Each `.txt` file contains lines formatted as:

```
relative/path/to/image.nii.gz  relative/path/to/label.nii.gz
```

## Pretrained Model Weights

| Model | `--model` | Source |
|-------|-----------|--------|
| Med3D | `Med3D` | [MedicalNet](https://github.com/Tencent/MedicalNet) |
| nnU-Net | `nnUNet` | [nnUNet](https://github.com/MIC-DKFZ/nnUNet) |
| UNETR | `UNETR` | [MONAI Model Zoo](https://monai.io/model-zoo.html) |
| Swin-UNETR | `SwinUNETR` | [MONAI Model Zoo](https://monai.io/model-zoo.html) |
| SAM-Med3D | `SAMMed3D` | [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D) |

## Usage

### Stage 1: Train a segmentation model

```bash
python train.py \
    --model Med3D \
    --dataset Kits21 \
    --data_root ./data \
    --n_epochs 200 \
    --learning_rate 0.001
```

### Stage 2: Generate probability maps

```bash
python inference.py \
    --model Med3D \
    --dataset Kits21 \
    --data_root ./data \
    --img_list ./data/ULS23/train.txt \
    --resume_path ./trials/Med3D/Kits21/epoch_best.pth.tar
```

### Stage 3: Conformal calibration

```bash
python calibration.py \
    --model Med3D \
    --dataset Kits21 \
    --data_root ./data \
    --risk_level 0.2 \
    --alpha 0.2 \
    --n_splits 100 \
    --cal_ratio 0.5
```

### Stage 4: Risk-controlled testing

```bash
python test.py \
    --model Med3D \
    --dataset Kits21 \
    --data_root ./data \
    --risk_level 0.2 \
    --alpha 0.2 \
    --n_splits 100 \
    --cal_ratio 0.5
```

## Reproducing Paper Results

**Table 1** (V-FNR and P-FNR on KiTS21 with varying $\varepsilon$):

```bash
for eps in 0.1 0.3 0.5; do
    python test.py --model Med3D --dataset Kits21 --risk_level $eps --alpha 0.2
done
```

**Figure 3** (ECR vs. $\alpha$ across all datasets):

```bash
for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    python test.py --model Med3D --dataset Kits21 --risk_level 0.4 --alpha $alpha
done
```

**Figure 5b** (ECR vs. calibration set proportion):

```bash
for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    python test.py --model Med3D --dataset Kits21 --risk_level 0.4 --alpha 0.2 --cal_ratio $ratio
done
```

## Citation

```bibtex
@inproceedings{tan2026conformal,
  title     = {Conformal 3D Lesion Segmentation with Balanced Risk Control},
  author    = {Tan, Binyu and Wang, Zhiyuan and Duan, Jinhao and Xu, Kaidi and
               Shen, Heng Tao and Shen, Fumin and Shi, Xiaoshuang},
  booktitle = {International Conference on Medical Image Computing and
               Computer-Assisted Intervention (MICCAI)},
  year      = {2026}
}
```

## Acknowledgments

The paper was supported by Noncommunicable Chronic Diseases-National Science and Technology Major Project (2025ZD0551300, 2025ZD0551302).

## License

This project is licensed under the MIT License.
