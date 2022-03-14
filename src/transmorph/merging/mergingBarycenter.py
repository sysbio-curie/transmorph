#!/usr/bin/env python3

import numpy as np

from .mergingABC import MergingABC

from ..matching.matchingABC import MatchingABC
from ..utils.anndata_interface import get_matrix

from anndata import AnnData
from typing import List


class MergingBarycenter(MergingABC):
    """
    Barycentric merging is the most naive way of merging datasets.
    Given n vectorized datasets x1, ..., xn and a reference dataset y,
    and n row-normalized matchings m1, ..., mn, we compute the following
    projection:

        f(xi) = mi @ y

    In other terms, every sample from xi is projected to the weighted sum
    of its matches in y. This works quite well with optimal transport-based
    methods, but is quite limited in the MNN case due to unmatched samples.

    Parameters
    ----------
    matching: MatchingABC
        Fitted, referenced matching between datasets.
    """

    def __init__(self):
        MergingABC.__init__(self, use_reference=True)

    def _check_input(self) -> None:
        """
        Raises an additional warning if some source samples are unmatched.
        TODO
        """
        super()._check_input()
        pass

    def fit(
        self,
        datasets: List[AnnData],
        matching: MatchingABC,
        X_kw: str,
        reference_idx: int = -1,
    ) -> List[np.ndarray]:
        assert reference_idx >= 0, "Missing reference dataset."
        representations = [get_matrix(adata, X_kw) for adata in datasets]
        adata_ref = datasets[reference_idx]
        X_ref = representations[reference_idx]
        output = []
        for k, adata in enumerate(datasets):
            if k == reference_idx:
                output.append(X_ref)
                continue
            T = matching.get_matching(adata, adata_ref, row_normalize=True).toarray()
            projection = T @ X_ref
            output.append(projection)
        return output
