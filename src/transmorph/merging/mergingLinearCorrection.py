#!/usr/bin/env python3

import numpy as np

from scipy.sparse import csr_matrix
from typing import List

from ..matching.matchingABC import MatchingABC
from .mergingABC import MergingABC


class MergingLinearCorrection(MergingABC):
    """ """

    def __init__(self, reference: np.ndarray):
        MergingABC.__init__(self, merge_on_reference=True)
        self.reference = reference

    def _check_input(self, datasets: List[np.ndarray], matching: MatchingABC) -> None:
        super()._check_input(datasets, matching)
        dim_reference = self.reference.shape[1]
        for k, dataset in enumerate(datasets):
            assert (
                dataset.shape[1] == dim_reference
            ), f"Error: dimension of dataset {k} inconsistent with reference."

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
