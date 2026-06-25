"""
KiTS21: Kidney Tumor Segmentation Challenge 2021.
Heller et al., arXiv:2307.01984, 2023.

Labels: 0=background, 1=kidney, 2=tumor, 3=cyst.
For CLS, the lesion class is tumor (label >= 1 binarized).
HU window: [-200, 300] for abdominal CT.
"""

import numpy as np
from benchmark import BaseSegDataset


class Kits21(BaseSegDataset):

    def _preprocess(self, img):
        img = np.clip(img, -200, 300)
        min_val, max_val = img.min(), img.max()
        if max_val - min_val > 0:
            img = (img - min_val) / (max_val - min_val)
        return img
