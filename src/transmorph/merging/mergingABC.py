#!/usr/bin/env python3
#
from abc import ABC, abstractmethod

from ..matching.matchingABC import MatchingABC

import numpy as np
from typing import List
from anndata import AnnData


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

    def __init__(self, use_reference: bool = False):
        self.use_reference = use_reference

    def _check_input(self) -> None:
        """
        Checking if number of matchings and datasets coincides with reference strategy.
        This method is automatically called at the beginning MergingABC._check_input().
        Any class inheriting from MergingABC can add rules to this method.
        """
        pass

    @abstractmethod
    def fit(
        self,
        datasets: List[AnnData],
        matching: MatchingABC,
        X_kw: str,
        reference_idx: int = -1,
    ) -> List[np.ndarray]:
        pass
