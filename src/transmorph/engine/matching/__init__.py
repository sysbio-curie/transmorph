#!/usr/bin/env python3

from scipy.sparse import csr_matrix
from typing import Dict, Tuple

from .algorithms.matchingot import MatchingOT
from .algorithms.matchingmnn import MatchingMNN
from .algorithms.matchinggw import MatchingGW
from .algorithms.matchingfusedgw import MatchingFusedGW
from .matching import Matching


# This is the low-level type of a matching set
# between datasets, we shortcut it as it is quite
# often used.
_TypeMatchingSet = Dict[Tuple[int, int], csr_matrix]


__all__ = [
    "_TypeMatchingSet",
    "Matching",
    "MatchingMNN",
    "MatchingGW",
    "MatchingFusedGW",
    "MatchingOT",
]
