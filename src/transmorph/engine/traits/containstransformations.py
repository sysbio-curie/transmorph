#!/usr/bin/env python3

import numpy as np

from anndata import AnnData
from typing import List

from . import (
    HasMetadata,
    IsRepresentable,
    UsesCommonFeatures,
    assert_trait,
)
from ..transforming import Transformation
from ...utils.type import assert_type


class ContainsTransformations:
    """
    This gives the ability to an object to contain internal transformations.
    """

    def __init__(self) -> None:
        self.transformations: List[Transformation] = []

    @property
    def has_transformations(self) -> bool:
        return len(self.transformations) > 0

    @property
    def preserves_space(self) -> bool:
        return all(tr.preserves_space for tr in self.transformations)

    def add_transformation(self, transformation: Transformation) -> None:
        """
        Adds a transformation step to the layer, that will be applied
        before running the internal algorithm.
        """
        assert_type(transformation, Transformation)
        assert (
            transformation not in self.transformations
        ), "Transformation already exists."
        self.transformations.append(transformation)

    def transform(
        self, datasets: List[AnnData], representer: IsRepresentable
    ) -> List[np.ndarray]:
        """
        Runs all transformations.
        """
        is_feature_space = representer.is_feature_space
        assert_trait(representer, IsRepresentable)
        Xs = [representer.get_representation(adata) for adata in datasets]
        for transformation in self.transformations:
            # If necessary, we let transformation retrieve
            # additional information
            if isinstance(transformation, HasMetadata):
                transformation.retrieve_all_metadata(datasets)
            if isinstance(transformation, UsesCommonFeatures):
                transformation.retrieve_common_features(datasets, is_feature_space)
            transformation.check_input(Xs)
            Xs = transformation.transform(Xs)
            is_feature_space = transformation.preserves_space
        return Xs
