"""
LIDC-IDRI: Lung Image Database Consortium.
Armato et al., Medical Physics, 2011.

Labels: 0=background, 1=nodule.
HU window: [-1000, 400] for lung CT.
"""

import numpy as np
from benchmark import BaseSegDataset


class LIDC(BaseSegDataset):

    def _preprocess(self, img):
        img = np.clip(img, -1000, 400)
        min_val, max_val = img.min(), img.max()
        if max_val - min_val > 0:
            img = (img - min_val) / (max_val - min_val)
        return img
