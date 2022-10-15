#!/usr/bin/env python3

import numpy as np

from anndata import AnnData
from typing import Callable, List, Optional

from .utils import assert_trait
from .isrepresentable import IsRepresentable
from ..transforming.transformation import Transformation
from ..traits.utils import preprocess_traits
from ...utils.misc import assert_type


class ContainsTransformations:
    """
    A class inheriting this trait can contain and run
    Transformation objects.

    Attributes
    ----------
    transformations: List[Transformation]
        A list of transformation objects, expected to be already
        parametrized, and equiped with a transform() method.
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
        before running the internal algorithm. Transformations will be
        applied in the order of addition.

        Parameters
        ----------
        transformation: Transformation
            Transformation object to append, already parametrized.
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
        Runs all transformations in the order of addition via add_transformation,
        then returns the final result.

        Parameters
        ----------
        datasets: List[AnnData]
            List of datasets represented as AnnData objects.

        representer: IsRepresentable
            Layer providing the embedding reference for these AnnData objects.

        log_callback: Optional[Callable]
            Logging function to use if necessary. If left None, won't log anything.
        """
        is_feature_space = representer.is_feature_space
        assert is_feature_space is not None
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
            preprocess_traits(transformation, datasets, is_feature_space)
            transformation.check_input(Xs)
            if log_callback is not None:
                init_dimension = f"[{', '.join([str(X.shape) for X in Xs])}]"
                log_callback(f"Initial spaces dimension: {init_dimension}")
            Xs = transformation.transform(datasets, Xs)
            is_feature_space = is_feature_space and transformation.preserves_space
            if log_callback is not None:
                final_dimension = f"[{', '.join([str(X.shape) for X in Xs])}]"
                log_callback(f"Final spaces dimension: {final_dimension}")
                log_callback(f"Is feature space: {is_feature_space}")
        return Xs
