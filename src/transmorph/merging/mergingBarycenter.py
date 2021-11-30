#!/usr/bin/env python3

import numpy as np

from scipy.sparse import csr_matrix
from typing import List

from ..matching.matchingABC import MatchingABC
from .mergingABC import MergingABC


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
    reference_index: int, default = 0
        Reference dataset index

    Return type
    -----------
    np.ndarray containing an embedding of all datasets in y's space.
    """

    def __init__(self, reference: np.ndarray, handle_unmatched: bool = False):
        MergingABC.__init__(self, merge_on_reference=True)
        self.reference = reference
        self.handle_unmatched = handle_unmatched

    def _check_input(self, datasets: List[np.ndarray], matching: MatchingABC) -> None:
        super()._check_input(datasets, matching)
        if self.handle_unmatched:
            return
        for k, dataset in enumerate(datasets):
            T = matching.get(k, normalize=True)
            if type(T) == csr_matrix:
                T = T.toarray()
            if any(T.sum(axis=1) == 0.0):
                print(
                    "Warning: Some samples are unmatched. "
                    "Try a different matching or merging strategy."
                )
                raise ValueError

    def merge(self, datasets: List[np.ndarray], matching: MatchingABC) -> np.ndarray:
        self._check_input(datasets, matching)
        ref_dataset = datasets[self.reference_index]
        output = np.zeros(
            (sum(dataset.shape[0] for dataset in datasets), ref_dataset.shape[1])
        )
        offset = 0
        for k, dataset in enumerate(datasets):
            n = dataset.shape[0]
            projection = ref_dataset
            if k == self.reference_index:
                continue
            T = matching.get(k, self.reference_index, normalize=True)
            if type(T) == csr_matrix:
                T = T.toarray()
            projection = T @ ref_dataset
            output[offset : offset + n] = projection
            offset += n
        return output
