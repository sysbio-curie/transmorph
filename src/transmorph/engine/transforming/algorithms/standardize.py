#!/usr/bin/env python3


import numpy as np

from typing import List

from .. import Transformation


class Standardize(Transformation):
    """
    Centers (substracts mean) and scales (divides by STD) datasets.

    Parameters
    ----------
    center: bool, default = True
        Applies the centering.

    scale: bool, default = True
        Applies the scaling.
    """

    def __init__(self, center: bool = True, scale: bool = True):
        Transformation.__init__(
            self, str_identifier="STANDARDIZE", preserves_space=True
        )
        self.center = center
        self.scale = scale

    def transform(self, datasets: List[np.ndarray]) -> List[np.ndarray]:
        results = []
        for X in datasets:
            X = X.copy()
            if self.center:
                X -= X.mean(axis=0)
            if self.scale:
                normalizer = np.std(X, axis=0)
                normalizer[normalizer == 0.0] = 1.0
                X /= normalizer
            results.append(X)
        return results
