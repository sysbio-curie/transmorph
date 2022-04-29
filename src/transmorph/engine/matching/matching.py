#!/usr/bin/env python3

from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple

from ..traits.canlog import CanLog
from ..traits.isprofilable import IsProfilable

# This is the low-level type of a matching set
# between datasets, we shortcut it as it is quite
# often used.
_TypeMatchingSet = Dict[Tuple[int, int], csr_matrix]


class Matching(ABC, IsProfilable, CanLog):
    """
    A Matching is an algorithm that is used to assess similarity between
    samples across datasets. Mergings use this information to build a
    common embedding between datasets. In our framework a matching between
    two datasets is represented as a sparse matrix whose value i,j represent
    matching strength between sample i from the first dataset and sample
    j from the second dataset.

    Every matching algorithm must inherit from the Matching class, and at
    least implement a fit() method. This method takes as input a list of
    matrices representing datasets to match, and returns a dictionary D
    where D[i, j] contains the sparse matching between datasets i and j.
    Access to other type of metadata can be granted by inheriting specific
    traits, and implementing their methods.
    """

    def __init__(
        self,
        str_identifier: str = "DEFAULT",
    ):
        CanLog.__init__(self, str_identifier=f"MATCHING_{str_identifier}")
        IsProfilable.__init__(self)

    def check_input(self, datasets: List[np.ndarray]) -> None:
        """
        Takes a list of np.ndarray representing datasets to match,
        and verifies their validity. Should raise warnings or
        errors in case of unexpected input. Will be called before
        carrying out the matching task. Can be overrode by child
        classes.
        """
        pass

    @abstractmethod
    def fit(self, datasets: List[np.ndarray]) -> _TypeMatchingSet:
        """
        Computes pairwise matchings between a set of numpy ndarrays, and
        returns these matchings as a dictionary of int tuples -> csr_matrix.
        Must be implemented in each Matching child class.

        Parameters:
        -----------
        datasets: List[np.ndarray]
            List of numpy ndarray datasets.
        """
        pass
