#!/usr/bin/env python3

import warnings
import numpy as np

from .mergingABC import MergingABC

from ..matching.matchingABC import MatchingABC
from ..utils.anndata_interface import get_matrix

from anndata import AnnData
from typing import List, Union
from scipy.sparse import csr_matrix, csc_matrix


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

    def _check_input(
        self,
        datasets: List[AnnData],
        matching: Union[MatchingABC, None],
        matching_mtx: Union[csr_matrix, np.ndarray, None],
        X_kw: str = "",
        reference_idx: int = -1,
    ) -> None:
        super()._check_input(datasets, matching, matching_mtx, X_kw, reference_idx)
        if matching_mtx is not None:
            if (
                type(matching_mtx) is not csr_matrix
                and type(matching_mtx) is not np.ndarray
            ):
                raise TypeError(f"Unrecognized matching type {type(matching_mtx)}")
            if any(matching_mtx.sum(axis=1) == 0):
                warnings.warn(
                    "Unmatched samples detected. You may want to switch to another "
                    "merging such as MergingLinearCorrection to avoid degenerate"
                    "solutions."
                )

    def fit(
        self,
        datasets: List[AnnData],
        matching: Union[MatchingABC, None] = None,
        matching_mtx: Union[csr_matrix, np.ndarray, None] = None,
        X_kw: str = "",
        reference_idx: int = -1,
    ) -> List[np.ndarray]:
        self._check_input(datasets, matching, matching_mtx, X_kw, reference_idx)
        representations = [get_matrix(adata, X_kw) for adata in datasets]
        adata_ref = datasets[reference_idx]
        X_ref = representations[reference_idx]
        output = []
        for k, adata in enumerate(datasets):
            if k == reference_idx:
                output.append(X_ref)
                continue
            if matching is not None:
                T = matching.get_matching(adata, adata_ref, row_normalize=True)
            elif matching_mtx is not None:  # Works as an elif
                T = matching_mtx
                if T.shape[0] == adata_ref.n_obs:
                    T = T.T
                if type(T) is csc_matrix or type(T) is csr_matrix:
                    T = T.toarray()
                assert type(T) is np.ndarray, f"Unrecognized type: {type(T)}"
                T = csr_matrix(T / T.sum(axis=1, keepdims=True))
            else:
                raise AssertionError("matching or matching_mtx must be set.")
            assert type(T) is csr_matrix, f"Unrecognized type: {type(T)}"
            projection = T @ X_ref
            output.append(projection)
        return output
