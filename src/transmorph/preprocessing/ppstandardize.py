#!/usr/bin/env python3

from typing import List
from anndata import AnnData

import numpy as np

from .preprocessingABC import PreprocessingABC
from ..utils.anndata_interface import get_matrix


class PPStandardize(PreprocessingABC):
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
        self.center = center
        self.scale = scale

    def transform(self, datasets: List[AnnData], X_kw: str = "") -> List[np.ndarray]:
        results = []
        for adata in datasets:
            X = get_matrix(adata, X_kw).copy()
            if self.center:
                X -= X.mean(axis=0)
            if self.scale:
                normalizer = np.std(X, axis=0)
                normalizer[normalizer == 0.0] = 1.0
                X /= normalizer
            results.append(X)
        return results
