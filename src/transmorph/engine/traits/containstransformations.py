#!/usr/bin/env python3

import numpy as np

from anndata import AnnData
from typing import Callable, List, Optional

from .utils import assert_trait
from .hasmetadata import HasMetadata
from .isrepresentable import IsRepresentable
from .usescommonfeatures import UsesCommonFeatures
from ..transforming.transformation import Transformation
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
        self,
        datasets: List[AnnData],
        representer: IsRepresentable,
        log_callback: Optional[Callable] = None,
    ) -> List[np.ndarray]:
        """
        Runs all transformations. A logging function can be passed as
        parameter to compensate for not inheriting from CanLog.
        """
        is_feature_space = representer.is_feature_space
        if log_callback is not None:
            log_callback(
                f"Beginning of transform(). Is feature space: {is_feature_space}"
            )
        assert_trait(representer, IsRepresentable)
        Xs = [representer.get_representation(adata) for adata in datasets]
        for transformation in self.transformations:
            # If necessary, we let transformation retrieve
            # additional information
            if log_callback is not None:
                log_callback(f"Running transformation {transformation}")
            if isinstance(transformation, HasMetadata):
                transformation.retrieve_all_metadata(datasets)
            if isinstance(transformation, UsesCommonFeatures):
                transformation.retrieve_common_features(datasets, is_feature_space)
            transformation.check_input(Xs)
            if log_callback is not None:
                init_dimension = f"[{', '.join([str(X.shape[1]) for X in Xs])}]"
                log_callback(f"Initial spaces dimension: {init_dimension}")
            Xs = transformation.transform(Xs)
            is_feature_space = is_feature_space and transformation.preserves_space
            if log_callback is not None:
                final_dimension = f"[{', '.join([str(X.shape[1]) for X in Xs])}]"
                log_callback(f"Final spaces dimension: {final_dimension}")
                log_callback(f"Is feature space: {is_feature_space}")
        return Xs
