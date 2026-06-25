"""
MDSC-Pancreas: Medical Segmentation Decathlon Task07 (Pancreas Tumor).
Antonelli et al., Nature Communications, 2022.

Labels: 0=background, 1=pancreas, 2=tumor.
HU window: [-100, 300] for abdominal CT.
"""

import numpy as np
from benchmark import BaseSegDataset


class PAN(BaseSegDataset):

    def _preprocess(self, img):
        img = np.clip(img, -100, 300)
        min_val, max_val = img.min(), img.max()
        if max_val - min_val > 0:
            img = (img - min_val) / (max_val - min_val)
        return img
