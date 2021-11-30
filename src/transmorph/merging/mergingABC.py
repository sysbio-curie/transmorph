#!/usr/bin/env python3
#
from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix

from ..matching.matchingABC import MatchingABC

import numpy as np
from typing import Union


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

    def __init__(self, matching: MatchingABC):
        self.use_reference = matching.get_reference() is not None

    def _check_input(self, matching: MatchingABC) -> None:
        """
        Checking if number of matchings and datasets coincides with reference strategy.
        This method is automatically called at the beginning MergingABC._check_input().
        Any class inheriting from MergingABC can add rules to this method.
        """
        n_datasets = matching.n_datasets
        assert n_datasets > 0, "Error: No datasets found for merging."
        if self.use_reference:
            assert n_datasets == matching.n_matchings, (
                "Error: Inconsistent number of matchings and datasets "
                f"for merging strategy using a reference. Found {n_datasets} "
                f"dataset(s) for {matching.n_matchings} matching(s)."
            )
        else:
            n_matchings = (n_datasets - 1) * (n_datasets - 2) / 2
            assert n_matchings == matching.n_matchings, (
                "Error: Inconsistent number of matchings and datasets "
                f"for merging strategy without reference. Found {n_datasets} "
                f"dataset(s) for {matching.n_matchings} matching(s) "
                f"(expected {n_matchings})."
            )

    @abstractmethod
    def transform(self, matching: MatchingABC) -> Union[np.ndarray, csr_matrix]:
        self._check_input(matching)
