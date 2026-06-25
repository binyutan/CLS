"""
MDSC-Colon: Medical Segmentation Decathlon Task10 (Colon Cancer).
Antonelli et al., Nature Communications, 2022.

Labels: 0=background, 1=colon cancer.
HU window: [-200, 400] for abdominal CT.
"""

import numpy as np
from benchmark import BaseSegDataset


class COLON(BaseSegDataset):

    def _preprocess(self, img):
        img = np.clip(img, -200, 400)
        min_val, max_val = img.min(), img.max()
        if max_val - min_val > 0:
            img = (img - min_val) / (max_val - min_val)
        return img
