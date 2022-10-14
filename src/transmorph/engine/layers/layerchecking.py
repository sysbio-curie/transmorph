#!/usr/bin/env python3

import logging

from anndata import AnnData
from typing import List, Optional

from . import Layer
from ..checking import Checking
from ..traits import (
    CanCatchChecking,
    ContainsTransformations,
    IsProfilable,
    profile_method,
    IsRepresentable,
    UsesSubsampling,
    assert_trait,
)
from ..traits.utils import preprocess_traits
from ..subsampling import Subsampling


class LayerChecking(
    Layer,
    IsProfilable,
    IsRepresentable,
    UsesSubsampling,
    ContainsTransformations,
    CanCatchChecking,
):
    """
    A checking layer is a layer that encapsulates a checking algorithm.
    It is used to create logic branchings in Models, and has two
    ouputs:

    - Forward output, containing the next pipeline step if the
      checking is accepted.

    - Rejected output, containing the fallback step to carry out
      if checking is rejected.

    A checking layer can be used to create iterative procedures in
    a model, playing the role of a while loop ("while the checking
    is rejected, go back to an earlier pipeline step"). Temporary
    transformations can be loaded in a checking layer to be carried
    out before the checking algorithm.

    Parameters
    ----------
    checking : Checking
        Checking algorithm contained in the layer. This object is
        endowed with a check() method, that will be called by the
        layer. Model execution will then continue according to
        the result.

    min_score_variation : float, default = 0.01
        Minimum score improvement between two checkings necessary
        to avoid early exit.

    n_checks_min : int, default = 3
        Minuimum number of checkings to be carried out by the layer
        before taking into account score improvement.

    n_checks_max : int, default = 10
        Maximum number of checkings to be carried out by the layer.
        Beyond this number, it will automatically accept to avoid
        endless looping.

    subsampling : Optional[Subsampling], default = None
        Subsampling algorithm to use before the checking, can help
        for performance when dealing with large datasets.
    """

    # TODO: remove IsRepresentable trait from LayerChecking
    #
    # Attributes
    # ----------
    # check_is_valid: Optional[bool]
    #     Contains validity of the last check, useful for logging
    #     purposes.

    # n_checks: int
    #     Number of checkings that have been carried out by the layer.

    # rejected_layer: Optional[CanCatchChecking]
    #     Reference to the rejected output. This output must be an
    #     instance of CanCatchChecking, meaning it is equipped to
    #     receive information of a LayerChecking in case of
    #     rejection.

    # rejected_layer_ref: Optional[IsRepresentable]
    #     Used to temporarily store the embedding reference of
    #     rejected layer, that will be swapped during the loop with
    #     this CheckingLayer.
    def __init__(
        self,
        checking: Checking,
        min_score_variation: float = 0.01,
        n_checks_min: int = 3,
        n_checks_max: int = 10,
        subsampling: Optional[Subsampling] = None,
    ) -> None:
        Layer.__init__(
            self,
            compatible_inputs=[IsRepresentable],
            str_identifier="CHECKING",
        )
        IsProfilable.__init__(self)
        IsRepresentable.__init__(self, repr_key=f"{self}#{self.layer_id}")
        UsesSubsampling.__init__(self, subsampling)
        ContainsTransformations.__init__(self)
        CanCatchChecking.__init__(self)
        self.check_is_valid: Optional[bool] = None
        self.checking = checking
        self.min_score_variation = min_score_variation
        self.n_checks_min = n_checks_min  # Min checks before early stop
        self.n_checks_max = n_checks_max  # Max checks allowed
        self.n_checks = 0  # Numbers of checkings done
        self.rejected_layer: Optional[CanCatchChecking] = None
        self.rejected_layer_ref: Optional[IsRepresentable] = None
        self.scores: List[float] = []

    def connect_rejected(self, layer: CanCatchChecking):
        """
        Connects this layer to another layer which will be called
        in case of rejection. Target layer can be upstream in the
        pipeline. It must inherit CanCatchChecking trait. A
        LayerChecking can only have one target rejected layer.

        Parameters
        ----------
        layer: CanCatchChecking
            Target layer to call in the rejected case.
        """
        assert_trait(layer, CanCatchChecking)
        assert isinstance(layer, Layer)
        layer.catch_checking_rejected(self)
        self.rejected_layer = layer

    @profile_method
    def fit(self, datasets: List[AnnData]) -> List[Layer]:
        """
        Runs the internal algorithm after carrying out the
        appropriate preprocessings. Then, returns either the
        rejected layer or standard output layers depending
        on checking result.

        Parameters
        ----------
        datasets: List[AnnData]
            Datasets to run checking on.
        """
        assert self.rejected_layer is not None, "A rejected layer must be specified."
        # Writing previous output for next layers to use
        self.log(f"Retrieving data from {self.embedding_reference.repr_key}.")
        Xs = [self.embedding_reference.get_representation(adata) for adata in datasets]
        is_feature_space = self.embedding_reference.is_feature_space
        assert is_feature_space is not None
        for adata, X in zip(datasets, Xs):
            self.write_representation(
                adata,
                X,
                is_feature_space,
            )
        # Preprocessing for self.checking if necessary
        if self.has_transformations:
            self.log("Calling preprocessings.", level=logging.INFO)
        Xs = self.transform(datasets, self.embedding_reference)
        # Subsampling if necessary
        self.compute_subsampling(
            datasets=datasets,
            matrices=Xs,
            is_feature_space=is_feature_space,
            log_callback=self.log,
        )
        Xs = self.subsample_matrices(Xs)
        # Retrieving metadata and common features if asked by
        # self.checking
        preprocess_traits(self.checking, datasets, is_feature_space)
        # Performing actual checking
        self.n_checks += 1
        check_passed = self.checking.check(Xs)

        self.scores.append(self.checking.score)

        insufficient_variation, variation = False, 0.0
        if len(self.scores) > 1:
            variation = abs(self.scores[-1] - self.scores[-2]) / self.scores[-2]
            insufficient_variation = variation < self.min_score_variation

        self.info(f"Checking score: {self.scores[-1]}")

        self.check_is_valid = (
            check_passed
            or insufficient_variation
            and self.n_checks >= self.n_checks_min
            or self.n_checks >= self.n_checks_max
        )

        # Routing accordingly
        if self.check_is_valid:
            if self.n_checks >= self.n_checks_max:
                self.info("Maximum number of checks reached. Continuing.")
            if insufficient_variation:
                self.info(
                    f"Insufficient improvement ({variation} < "
                    f"{self.min_score_variation}). Continuing."
                )
            return self.output_layers
        else:
            self.log("Checking failed. Continuing.", level=logging.INFO)
            self.rejected_layer.called_by_checking = True
            assert isinstance(self.rejected_layer, Layer)
            return [self.rejected_layer]
