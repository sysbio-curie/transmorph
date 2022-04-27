#!/usr/bin/env python3

import logging

from anndata import AnnData
from typing import List, Optional

from . import Layer
from ..checking import Checking
from ..traits import (
    CanCatchChecking,
    ContainsTransformations,
    HasMetadata,
    IsProfilable,
    profile_method,
    IsRepresentable,
    IsSubsamplable,
    UsesCommonFeatures,
    assert_trait,
)
from ..subsampling import Subsampling


class LayerChecking(
    Layer,
    IsProfilable,
    IsRepresentable,
    IsSubsamplable,
    ContainsTransformations,
    CanCatchChecking,
):
    """
    A LayerChecking is a complex layer with two types of outputs, that
    encapsulates a Checking algorithm. It is used to create logic
    branchings in Models. The two type of outputs are

    > Standard outputs, that represent next pipeline steps if the
      checking is accepted.

    > Rejected output, must be unique. It references a layer to
      call instead of the standard ones in case the checking is
      rejected.

    A LayerChecking can be used to create iterative procedures in
    a model, playing the role of a while loop ("while the checking
    is rejected, go back to earlier in the pipeline and apply the
    integration procedure"). Temporary transformations can be loaded
    in LayerMatching to be carried out before the matching algorithm.

    Parameters
    ----------
    checking: Checking
        Checking algorithm contained in the layer. This object is
        endowed with a check() method, that will be called by the
        layer. Model execution will then continue according to
        the result.

    n_checks_max: int, default = 10
        Maximum number of checkings to be carried out by the layer.
        Beyond this number, it will automatically accept to avoid
        endless looping.

    subsampling: Optional[Subsampling], default = None
        Subsampling algorithm to use before the checking, can help
        for performance when dealing with large datasets.

    Attributes
    ----------
    check_is_valid: Optional[bool]
        Contains validity of the last check, useful for logging
        purposes.

    n_checks: int
        Number of checkings that have been carried out by the layer.

    rejected_layer: Optional[CanCatchChecking]
        Reference to the rejected output. This output must be an
        instance of CanCatchChecking, meaning it is equipped to
        receive information of a LayerChecking in case of
        rejection.

    rejected_layer_ref: Optional[IsRepresentable]
        Used to temporarily store the embedding reference of
        rejected layer, that will be swapped during the loop with
        this CheckingLayer.
    """

    def __init__(
        self,
        checking: Checking,
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
        IsSubsamplable.__init__(self, subsampling)
        ContainsTransformations.__init__(self)
        CanCatchChecking.__init__(self)
        self.check_is_valid: Optional[bool] = None
        self.checking = checking
        self.n_checks_max = n_checks_max  # Max checks allowed
        self.n_checks = 0  # Numbers of checkings done
        self.rejected_layer: Optional[CanCatchChecking] = None
        self.rejected_layer_ref: Optional[IsRepresentable] = None

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
        Xs = [self.embedding_reference.get_representation(adata) for adata in datasets]
        is_feature_space = self.embedding_reference.is_feature_space
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
        self.subsample(datasets=datasets, matrices=Xs)
        Xs = self.slice_matrices(datasets=datasets, matrices=Xs)
        # Retrieving metadata and common features if asked by
        # self.checking
        if isinstance(self.checking, HasMetadata):
            self.checking.retrieve_all_metadata(datasets)
        if isinstance(self.checking, UsesCommonFeatures):
            self.checking.retrieve_common_features(datasets, is_feature_space)
        # Performing actual checking
        self.n_checks += 1
        self.check_is_valid = self.n_checks >= self.n_checks_max or self.checking.check(
            Xs
        )
        # Routing accordingly
        if self.check_is_valid:
            if self.n_checks >= self.n_checks_max:
                self.log("Maximum number of checks reached.", level=logging.INFO)
            return self.output_layers
        else:
            self.log("Checking failed. Continuing.", level=logging.INFO)
            self.rejected_layer.called_by_checking = True
            assert isinstance(self.rejected_layer, Layer)
            return [self.rejected_layer]
