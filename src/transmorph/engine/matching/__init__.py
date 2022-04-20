#!/usr/bin/env python3

from .algorithms.matchingot import MatchingOT
from .algorithms.matchingmnn import MatchingMNN
from .algorithms.matchinggw import MatchingGW
from .algorithms.matchingfusedgw import MatchingFusedGW
from .layermatching import LayerMatching
from .matching import Matching
from .watchermatching import WatcherMatching

__all__ = [
    "Matching",
    "MatchingMNN",
    "MatchingGW",
    "MatchingFusedGW",
    "MatchingOT",
    "LayerMatching",
    "WatcherMatching",
]
