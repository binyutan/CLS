"""
NIH-LN ABD: Abdominal Lymph Node Dataset.
Roth et al., MICCAI, 2014.

Labels: 0=background, 1=lymph node.
HU window: [-100, 300] for abdominal CT.
"""

import numpy as np
from benchmark import BaseSegDataset


class ABD(BaseSegDataset):

    def _preprocess(self, img):
        img = np.clip(img, -100, 300)
        min_val, max_val = img.min(), img.max()
        if max_val - min_val > 0:
            img = (img - min_val) / (max_val - min_val)
        return img
