#!/usr/bin/env python3

from .algorithms.bknn import BKNN
from .algorithms.combinematching import CombineMatching
from .algorithms.fusedgw import FusedGW
from .algorithms.gw import GW
from .algorithms.labels import Labels
from .algorithms.mnn import MNN
from .algorithms.ot import OT
from .matching import Matching, _TypeMatchingSet

__all__ = [
    "_TypeMatchingSet",
    "BKNN",
    "CombineMatching",
    "FusedGW",
    "GW",
    "Labels",
    "Matching",
    "MNN",
    "OT",
]
