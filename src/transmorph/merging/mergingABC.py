#!/usr/bin/env python3
#
from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix

from ..matching.matchingABC import MatchingABC

import numpy as np
from typing import Union, List


class MergingABC(ABC):
    """
    A Merging is a class containing a function merge(x1, ..., xn, m1, ..., mk)
    with n >= 2 and k >= 1, where x1, ..., xn are n datasets and m(i(i-1)/2+j)
    contains the matching between xi and xj. merge() outputs a common representation
    for all samples from datasets x1 to xn that can be an embedding or a network.

    Parameters
    ----------
    merge_on_reference: bool
        Do the merging use a reference dataset or is it symmetrical?

    reference: np.ndarray, default = None
        If merge_on_reference=True, reference dataset.
    """

    @abstractmethod
    def __init__(self, merge_on_reference, reference: np.ndarray = None):
        self.merge_on_reference = merge_on_reference
        self.reference = reference

    @abstractmethod
    def _check_input(self, datasets: List[np.ndarray], matching: MatchingABC) -> None:
        pass

    @abstractmethod
    def merge(
        self, datasets: List[np.ndarray], matching: MatchingABC
    ) -> Union[np.ndarray, csr_matrix]:
        self._check_input(datasets, matching)
        pass
