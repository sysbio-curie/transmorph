#!/usr/bin/env python3

# A trait is a small module of features that can be added
# to a class using inheritance. It allows code factorization,
# and easier compatibility checking. The base trait does nothing
# but checking if an object is endowed with it.

from .cancatchchecking import CanCatchChecking
from .canlog import CanLog
from .containstransformations import ContainsTransformations
from .hasmetadata import HasMetadata
from .isprofilable import IsProfilable, profile_method
from .isrepresentable import IsRepresentable
from .issubsamplable import IsSubsamplable
from .usescommonfeatures import UsesCommonFeatures
from .usesmatching import UsesMatching, _TypeMatchingModes
from .usesmetric import UsesMetric
from .usesneighbors import UsesNeighbors
from .usesreference import UsesReference
from .utils import assert_trait

__all__ = [
    "CanCatchChecking",
    "CanLog",
    "ContainsTransformations",
    "HasMetadata",
    "IsProfilable",
    "profile_method",
    "IsRepresentable",
    "IsSubsamplable",
    "UsesCommonFeatures",
    "UsesMatching",
    "_TypeMatchingModes",
    "UsesMetric",
    "UsesNeighbors",
    "UsesReference",
    "assert_trait",
]