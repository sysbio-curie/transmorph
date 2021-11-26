#!/usr/bin/env python3

from ..matching.matchingABC import MatchingABC
from .mergingABC import MergingABC
from scipy.sparse import csr_matrix
from typing import (
    List
)

import numpy as np


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
    def __init__(
            self,
            reference_index: int = 0
    ):
        self.reference_index = reference_index 


    def merge(
            self,
            datasets: List[np.ndarray],
            matching: MatchingABC
    ) -> np.ndarray:
        ref_dataset = datasets[self.reference_index]
        output = np.zeros((
            sum(dataset.shape[0] for dataset in datasets),
            ref_dataset.shape[1]
        ))
        offset = 0
        for k, dataset in enumerate(datasets):
            n = dataset.shape[0]
            projection = ref_dataset
            if k != self.reference_index:
                T = matching.get(k, self.reference_index, normalize=True)
                if type(T) == csr_matrix:
                    T = T.toarray()
                projection = T @ ref_dataset
            output[offset:offset+n] = projection
            offset += n
        return output
