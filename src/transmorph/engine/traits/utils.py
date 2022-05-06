#!/usr/bin/env python3

from anndata import AnnData
from typing import Any, List, Tuple, Type, Union


from .hasmetadata import HasMetadata
from .usescommonfeatures import UsesCommonFeatures
from .usesmetric import UsesMetric
from .usesreference import UsesReference
from .usessamplelabels import UsesSampleLabels


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


def preprocess_traits(
    obj: Any,
    datasets: List[AnnData],
    is_feature_space: bool,
) -> None:
    """
    Helper function to ensure all traits are preprocessed.
    """
    if isinstance(obj, HasMetadata):
        obj.retrieve_all_metadata(datasets)
    if isinstance(obj, UsesCommonFeatures):
        obj.retrieve_common_features(datasets, is_feature_space)
    if isinstance(obj, UsesMetric):
        obj.retrieve_all_metrics(datasets)
    if isinstance(obj, UsesReference):
        obj.retrieve_reference_index(datasets)
    if isinstance(obj, UsesSampleLabels):
        obj.retrieve_all_labels(datasets)
