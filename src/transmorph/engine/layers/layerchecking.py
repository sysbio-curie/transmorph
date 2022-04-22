#!/usr/bin/env python3

import logging

from anndata import AnnData
from typing import List, Optional

from . import Layer
from ..checking import Checking
from ..traits import CanCatchChecking, IsSubsamplable, ContainsTransformations
from ..subsampling import KeepAll
from ..traits import (
    HasMetadata,
    IsProfilable,
    IsRepresentable,
    UsesCommonFeatures,
    assert_trait,
)
from ..watchers import IsWatchable, WatcherTiming
from ... import profile_method


class LayerChecking(
    Layer,
    IsWatchable,
    IsProfilable,
    IsRepresentable,
    IsSubsamplable,
    ContainsTransformations,
    CanCatchChecking,
):
    """
    Conditional layers with exactly two outputs. Performs a statistical test
    on its input data (typically the result of a merging), then
    > if accepted, calls output_layers as other layers
    > if rejected, calls rejected_layer that must be equipped with CanCatchChecking
      (possibly upstream)
    Useful to create "until convergence" loops. Encapsulates a Checking module.
    """

    def __init__(
        self,
        checking: Checking,
        n_checks_max: int = 10,
    ) -> None:
        Layer.__init__(
            self,
            compatible_inputs=[IsRepresentable],
            str_identifier="CHECKING",
        )
        IsWatchable.__init__(self, compatible_watchers=[WatcherTiming])
        IsProfilable.__init__(self)
        IsRepresentable.__init__(self, repr_key=f"{self}#{self.layer_id}")
        self.checking = checking
        self.n_checks = 0  # Numbers of checkings done
        self.n_checks_max = n_checks_max  # Max checks allowed
        self.rejected_layer: Optional[CanCatchChecking] = None
        self.rejected_layer_ref: Optional[IsRepresentable] = None

    def connect_rejected(self, layer: CanCatchChecking):
        """
        Sets up the rejected connection.
        """
        assert_trait(layer, CanCatchChecking)
        assert isinstance(self.rejected_layer, Layer)
        layer.connect_rejected(self)
        self.rejected_layer = layer

    @profile_method
    def fit(self, datasets: List[AnnData]) -> List[Layer]:
        """
        Runs preprocessings if needed, then calls the internal
        checking. If valid, proceeds to output_layers. Otherwise,
        falls back to rejected_layer.
        """
        assert self.rejected_layer is not None, "A rejected layer must be specified."
        # Writing previous output for next layers
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
        if not isinstance(self.subsampling, KeepAll):
            self.log("Calling subsampling.", level=logging.INFO)
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
        is_valid = self.n_checks >= self.n_checks_max or self.checking.check(Xs)
        # Routing accordingly
        if is_valid:
            if self.n_checks >= self.n_checks_max:
                self.log("Maximum number of checks reached.", level=logging.INFO)
            self.log("Checking passed. Continuing.", level=logging.INFO)
            return self.output_layers
        else:
            self.log("Checking failed. Continuing.", level=logging.INFO)
            self.rejected_layer.called_by_checking = True
            assert isinstance(self.rejected_layer, Layer)
            return [self.rejected_layer]
