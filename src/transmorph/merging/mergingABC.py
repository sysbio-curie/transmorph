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
    matching: MatchingABC
        Fitted, referenced matching between datasets.
    """

    def __init__(self, matching: MatchingABC):
        self.matching = matching
        assert matching is not None, "self.matching cannot be None."
        self.use_reference = matching.use_reference
        self._check_input()

    def _check_input(self) -> None:
        """
        Checking if number of matchings and datasets coincides with reference strategy.
        This method is automatically called at the beginning MergingABC._check_input().
        Any class inheriting from MergingABC can add rules to this method.
        """
        matching = self.matching
        assert matching is not None
        assert matching.fitted, (
            "Matching is unfitted. Calling first the"
            " method Matching.fit([datasets], reference=None) is necessary."
        )
        n_datasets = matching.n_datasets
        n_matchings = matching.n_matchings
        assert n_datasets > 0, "Error: No datasets found for merging."
        valid_n_datasets = False
        if self.use_reference and n_datasets == n_matchings:
            valid_n_datasets = True
        if not valid_n_datasets:
            valid_n_datasets = n_matchings == ((n_datasets) * (n_datasets - 1) / 2)
        if not valid_n_datasets:
            if self.use_reference:
                raise ValueError(
                    "Error: Inconsistent number of matchings and datasets "
                    f"for merging strategy using a reference. Found {n_datasets} "
                    f"dataset(s) for {n_matchings} matching(s). Expected "
                    f"{n_datasets} or {n_datasets * (n_datasets - 1) / 2} "
                    f"matchings."
                )
            else:
                raise ValueError(
                    "Error: Inconsistent number of matchings and datasets "
                    f"for merging strategy without reference. Found {n_datasets} "
                    f"dataset(s) for {n_matchings} matching(s) "
                    f"(expected {n_datasets * (n_datasets - 1) / 2})."
                )

    @abstractmethod
    def transform(self) -> Union[np.ndarray, csr_matrix]:
        pass
