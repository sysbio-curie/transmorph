#!/usr/bin/env python3

import anndata as ad
import numpy as np

from typing import List

from ..transformation import Transformation
from ....utils.matrix import center_matrix, scale_matrix


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

    def transform(
        self,
        datasets: List[ad.AnnData],
        embeddings: List[np.ndarray],
    ) -> List[np.ndarray]:
        results = []
        for X in embeddings:
            X = X.copy()
            if self.center:
                X = center_matrix(X, axis=0)
            if self.scale:
                X = scale_matrix(X, axis=0)
            results.append(X)
        return results
