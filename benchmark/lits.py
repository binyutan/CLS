"""
LiTS: Liver Tumor Segmentation Benchmark.
Bilic et al., Medical Image Analysis, 2023.

Labels: 0=background, 1=liver, 2=lesion.
HU window: [-100, 400] for liver CT.
"""

import numpy as np
from benchmark import BaseSegDataset


class LITS(BaseSegDataset):

    def _preprocess(self, img):
        img = np.clip(img, -100, 400)
        min_val, max_val = img.min(), img.max()
        if max_val - min_val > 0:
            img = (img - min_val) / (max_val - min_val)
        return img
