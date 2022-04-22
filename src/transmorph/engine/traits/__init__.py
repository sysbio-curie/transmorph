#!/usr/bin/env python3

# A trait is a small module of features that can be added
# to a class using inheritance. It allows code factorization,
# and easier compatibility checking. The base trait does nothing
# but checking if an object is endowed with it.

from typing import Any, Tuple, Type, Union

from .cancatchchecking import CanCatchChecking
from .canlog import CanLog
from .containstransformations import ContainsTransformations
from .hasmetadata import HasMetadata
from .isprofilable import IsProfilable
from .isrepresentable import IsRepresentable
from .issubsamplable import IsSubsamplable
from .iswatchable import IsWatchable
from .usescommonfeatures import UsesCommonFeatures
from .usesmatching import UsesMatching, _TypeMatchingModes
from .usesmetric import UsesMetric
from .usesneighbors import UsesNeighbors
from .usesreference import UsesReference


def assert_trait(obj: Any, traits: Union[Type, Tuple[Type, ...]]):
    """
    Raises an exception if $obj is not endowed with the
    trait $trait.
    """
    if isinstance(obj, traits):
        return
    if isinstance(traits, Type):
        all_traits: str = traits.__name__
    else:
        all_traits: str = ", ".join([trait.__name__ for trait in traits])
    raise TypeError(
        f"Object {obj} of type {type(obj)} is not endowed"
        f" with trait(s) {all_traits}."
    )


__all__ = [
    "CanCatchChecking",
    "CanLog",
    "ContainsTransformations",
    "HasMetadata",
    "IsProfilable",
    "IsRepresentable",
    "IsSubsamplable",
    "IsWatchable",
    "UsesCommonFeatures",
    "UsesMatching",
    "_TypeMatchingModes",
    "UsesMetric",
    "UsesNeighbors",
    "UsesReference",
    "assert_trait",
]
