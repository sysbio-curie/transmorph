#!/usr/bin/env python3

from __future__ import annotations

import anndata as ad
import numpy as np

from abc import ABC, abstractmethod
from numpy import testing
from typing import List

from ..traits.canlog import CanLog
from ..traits.isprofilable import IsProfilable
from ...utils.misc import assert_type


class Transformation(ABC, CanLog, IsProfilable):
    """
    Abstract class for Preprocessing objects. Implements a method
    transform(List[np.ndarray]) that computes the preprocessing.
    Child classes can be enriched by traits.
    """

    def __init__(
        self,
        str_identifier: str = "DEFAULT",
        preserves_space: bool = False,
        transformation_rate: float = 1.0,
    ):
        CanLog.__init__(self, str_identifier=f"TRANSFORMATION_{str_identifier}")
        self.preserves_space = preserves_space
        if transformation_rate != 1.0:
            assert self.preserves_space, (
                "Partial transformations are only allowed for "
                "space-preserving transformations."
            )
        assert (
            0.0 <= transformation_rate <= 1.0
        ), f"Partial rate must be between 0.0 and 1.0, found {transformation_rate}."
        self.transformation_rate = transformation_rate

    def check_input(self, datasets: List[np.ndarray]) -> None:
        """
        Can be implemented to perform checks on datasets, and raise warnings
        or exceptions in case of issues.
        """
        pass

    @abstractmethod
    def transform(
        self,
        datasets: List[ad.AnnData],
        embeddings: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Takes a list of representations as input, and returns a list of
        representations as output in the same order. Retrieved metadata
        can be used in this step.
        """
        pass

    @staticmethod
    def assert_transform_equals(
        transformation: Transformation,
        datasets: List[ad.AnnData],
        embeddings: List[np.ndarray],
        targets: List[np.ndarray],
        decimal: float = 6.0,
        print_debug: bool = False,
    ) -> None:
        """
        For testing purposes, checks if a Transformation works as
        expected.

        Parameters
        ----------
        transformation: Transformation
            Transformation to test.

        datasets: List[np.ndrray]
            List of testing matrices

        targets: List[np.ndarray]
            List of expected values

        decimal: float, default = 6
            Number of decimals to use for assert_array_almost_equal

        print_debug: bool, default = False
            Prints additional debug information in console.
        """
        if print_debug:
            print(f"> transformation: {transformation}")
            print(f"> n_datasets: {len(datasets)}")
            print(f"> n_targets: {len(targets)}")
            print(f"> decimal: {decimal}")
        assert_type(transformation, Transformation)
        # To ensure transform does not change initial matrices
        initial_Xs = [X.copy() for X in embeddings]
        transformed_Xs = transformation.transform(
            datasets=datasets,
            embeddings=initial_Xs,
        )
        for X_tra, X_tar in zip(transformed_Xs, targets):
            # Transformed are correct
            if print_debug:
                print(f"> X transform: {type(X_tra)}, {X_tra.shape}, {X_tra.dtype}")
                print(f"> X target: {type(X_tar)}, {X_tar.shape}, {X_tar.dtype}")
            testing.assert_array_almost_equal(X_tra, X_tar, decimal=decimal)
        for X_ini, X_dat in zip(initial_Xs, embeddings):
            # Initial datasets unchanged
            testing.assert_array_equal(X_ini, X_dat)
            if print_debug:
                print(f"> X initial: {type(X_ini)}, {X_ini.shape}, {X_ini.dtype}")
                print(f"> X final: {type(X_dat)}, {X_dat.shape}, {X_dat.dtype}")
