from .matchingEMD import MatchingEMD
from .matchingCombined import MatchingCombined
from .matchingMNN import MatchingMNN
from .matchingGW import MatchingGW
from .matchingGWEntropic import MatchingGWEntropic
from .matchingPartialOT import MatchingPartialOT
from .matchingSinkhorn import MatchingSinkhorn
from .matchingFusedGW import MatchingFusedGW

__all__ = [
    "MatchingEMD",
    "MatchingCombined",
    "MatchingMNN",
    "MatchingGW",
    "MatchingGWEntropic",
    "MatchingPartialOT",
    "MatchingSinkhorn",
    "MatchingFusedGW",
]
