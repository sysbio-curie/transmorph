#!/usr/bin/env python3
#
from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix

from ..matching.matchingABC import MatchingABC

import numpy as np
from typing import (
    Union,
    List
)

class MergingABC(ABC):
    """
    A Merging is a class containing a function merge(x1, ..., xn, m1, ..., mk)
    with n >= 2 and k >= 1, where x1, ..., xn are n datasets and m(i(i-1)/2+j)
    contains the matching between xi and xj. merge() outputs a common representation
    for all samples from datasets x1 to xn that can be an embedding or a network.
    """
    @abstractmethod
    def __init__(
            self
    ):
        pass

    @abstractmethod
    def merge(
            self,
            datasets: List[np.ndarray],
            matching: MatchingABC
    ) -> Union[np.ndarray, csr_matrix]:
        pass