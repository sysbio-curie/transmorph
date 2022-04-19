#!/usr/bin/env python3

# All types are string by default
from __future__ import annotations

import logging

from abc import abstractmethod
from anndata import AnnData
from scipy.sparse import csr_matrix
from transmorph import logger
from transmorph.utils import anndata_manager as adm
from transmorph.utils import AnnDataKeyIdentifiers
from typing import List, Optional, Type

from ..checking.checkingABC import CheckingABC
from ..matching.matchingABC import MatchingABC
from ..merging.mergingABC import MergingABC
from ..utils.type import assert_type

from .profiler import IsProfilable, profile_method
from .traits import (
    IsRepresentable,
    IsPreprocessable,
    HasMetadata,
    assert_trait,
)
from .watchers import IsWatchable, WatcherMatching, WatcherTiming


class Layer:
    """
    A Layer wraps an integration module, and manage its connections
    with other modules. All Layers derive from this class, and can be
    enriched using traits.
    """

    # Provides a unique ID to each layer
    LayerID = 0

    def __init__(
        self,
        compatible_inputs: List[Type] = [],
        str_type: str = "BASE",
    ) -> None:
        self.compatible_inputs = compatible_inputs
        self.input_layer: Optional[Layer] = None
        self.output_layers: List[Layer] = []
        self.profiler = None
        self._embedding_reference = None
        self.time_elapsed = -1
        self.layer_id = Layer.LayerID
        self._str = f"{str_type}#{self.layer_id}"
        Layer.LayerID += 1
        self._log("Initialized.")

    def _log(self, msg: str, level: int = logging.DEBUG) -> None:
        """
        Transmits a message to the logging module.

        Parameters
        ----------
        msg: str
            Message to print

        leve: int, default = logging.DEBUG
            Message priority. Set it higher to make it pass filters.
        """
        logger.log(level, f"{self._str} > {msg}")

    def connect(self, layer: Layer) -> None:
        """
        Connects the current layer to an output layer, if compatible.

        Parameters
        ----------
        layer: Layer
            Output layer of compatible type.
        """
        assert_type(layer, Layer)
        assert_type(self, tuple(layer.compatible_inputs))
        assert layer not in self.output_layers, f"{self} already connected to {layer}."
        assert layer.input_layer is None, f"{layer} has already a predecessor."
        layer.input_layer = self
        self.output_layers.append(layer)
        self._log(f"Connected to layer {layer}.")
        if layer.embedding_reference is None:
            if not isinstance(self, IsRepresentable):
                reference = self.embedding_reference
            else:
                reference = self
            self._log(f"{reference} chosen as default embedding reference for {self}.")
            layer.embedding_reference = reference

    @abstractmethod
    def fit(self, datasets: List[AnnData]) -> List[Layer]:
        """
        This is the computational method, running an internal module.
        It returns a list of downstream layers, to call next.

        Parameters
        ----------
        datasets: List[AnnData]
            List of AnnData datasets to process.
        """
        pass

    @property
    def embedding_reference(self) -> IsRepresentable:
        """
        Retrieves closest Representable object upstream from current layer.
        """
        if self.embedding_reference is None:
            if self.input_layer is None:
                raise ValueError(
                    "Input layer is None. Please make sure the "
                    "pipeline contains at least an input layer."
                )
            self._embedding_reference = self.input_layer.embedding_reference
        return self.embedding_reference

    @embedding_reference.setter
    def embedding_reference(self, reference: IsRepresentable) -> None:
        """
        Sets a Representable object to be the one providing matrix
        representations of datasets.
        """
        assert_trait(reference, IsRepresentable)
        self._embedding_reference = reference


class LayerInput(Layer, IsRepresentable):
    """
    Every pipeline must contain exactly one input layer, followed by an
    arbitrary network structure. Every pipeline is initialized using this
    input layer.
    """

    def __init__(self) -> None:
        Layer.__init__(self, compatible_inputs=[], str_type="INPUT")
        IsRepresentable.__init__(
            self, repr_key=AnnDataKeyIdentifiers.BaseRepresentation
        )

    def fit(self, datasets: List[AnnData]) -> List[Layer]:
        """
        Simply calls the downstream layers.
        """
        self._log("Checking all representations are present.")
        for adata in datasets:
            assert (
                adm.get_value(adata, self.repr_key) is not None
            ), f"Representation {self.repr_key} missing in {adata}."
        self._log("All representations found. Continuing.")
        return self.output_layers


class LayerOutput(Layer, IsRepresentable):
    """
    Simple layer to manage network outputs. There cannot be several output layers.
    for now, but it is a TODO
    """

    def __init__(self) -> None:
        Layer.__init__(
            self,
            compatible_inputs=[IsRepresentable],
            str_type="OUTPUT",
        )
        IsRepresentable.__init__(
            self, repr_key=AnnDataKeyIdentifiers.BaseRepresentation
        )

    def fit(self, datasets: List[AnnData]) -> List[Layer]:
        """
        Simply retrieves latest data representation, and stores it
        under obsm["transmorph"] key.
        """
        for adata in datasets:
            X = self.embedding_reference.get(adata)
            self.write(adata, X)
        return []


class LayerMatching(Layer, IsPreprocessable, IsWatchable, IsProfilable):
    """
    This layer performs a matching between two or more datasets.
    It wraps an object derived from MatchingABC.
    """

    def __init__(self, matching: MatchingABC) -> None:
        Layer.__init__(
            self,
            compatible_inputs=[IsRepresentable],
            str_type="MATCHING",
        )
        IsPreprocessable.__init__(self)
        IsWatchable.__init__(self, compatible_watchers=[WatcherMatching, WatcherTiming])
        IsProfilable.__init__(self)
        self.matching = matching

    @profile_method
    def fit(self, datasets: List[AnnData]) -> List[Layer]:
        """
        Calling self.matching.fit after carrying out the requested
        preprocessings.
        """
        if self.has_preprocessings:
            self._log("Calling preprocessings.", level=logging.INFO)
        Xs = self.preprocess(datasets, self.embedding_reference)
        self._log("Calling matching.", level=logging.INFO)
        if isinstance(self.matching, HasMetadata):
            self.matching.retrieve_all_metadata(datasets)
        self.matching.fit(Xs)
        self._log("Fitted.", level=logging.INFO)
        return self.output_layers

    def get_matching(self, adata1: AnnData, adata2: AnnData) -> csr_matrix:
        return self.matching.get_matching(adata1, adata2)


class LayerMerging(Layer, IsPreprocessable, IsWatchable, IsProfilable, IsRepresentable):
    """
    This layer performs a merging between two or more datasets and their matchings.
    It wraps an object derived from MergingABC.
    """

    def __init__(self, merging: MergingABC) -> None:
        Layer.__init__(self, compatible_inputs=[LayerMatching], str_type="MERGING")
        IsPreprocessable.__init__(self)
        IsWatchable.__init__(self, compatible_watchers=[WatcherTiming])
        IsProfilable.__init__(self)
        IsRepresentable.__init__(self, repr_key=f"{self}_{self.layer_id}")
        self.merging = merging

    @profile_method
    def fit(self, datasets: List[AnnData]) -> List[Layer]:
        """
        Runs preprocessings, then delegate to the internal merging.
        """
        # Pleases the type checker
        assert isinstance(self.input_layer, LayerMatching)
        if self.has_preprocessings:
            self._log("Calling preprocessings.", level=logging.INFO)
        Xs = self.preprocess(datasets, self.embedding_reference)
        self._log("Running merging...", level=logging.INFO)
        if isinstance(self.merging, HasMetadata):
            self.merging.retrieve_all_metadata(datasets)
        Xs_transform = self.merging.fit(
            Xs,
            matching=self.input_layer.matching,
        )
        for adata, X_after in zip(datasets, Xs_transform):
            self.write(adata, X_after)
        self._log("Fitted.", level=logging.INFO)
        return self.output_layers


class LayerPreprocessing(
    Layer, IsPreprocessable, IsWatchable, IsProfilable, IsRepresentable
):
    """
    This layer encapsulates a series of preprocessing algorithms derived
    from PreprocessingABC.
    """

    def __init__(self) -> None:
        Layer.__init__(
            self,
            compatible_inputs=[IsRepresentable],
            str_type="PREPROCESSING",
        )
        IsPreprocessable.__init__(self)
        IsWatchable.__init__(self, compatible_watchers=[WatcherTiming])
        IsProfilable.__init__(self)
        IsRepresentable.__init__(self, repr_key=f"{self}_{self.layer_id}")

    @profile_method
    def fit(self, datasets: List[AnnData]) -> List[Layer]:
        """
        Simply runs preprocessing algorithms and returns the result.
        """
        if self.has_preprocessings:
            self._log("Calling preprocessings.", level=logging.INFO)
        Xs = self.preprocess(datasets, self.embedding_reference)
        for adata, X_after in zip(datasets, Xs):
            self.write(adata, X_after)
        self._log("Done.", level=logging.INFO)
        return self.output_layers


class LayerChecking(Layer, IsWatchable, IsProfilable, IsRepresentable):
    """
    Conditional layers with exactly two outputs. Performs a statistical test
    on its input data (typically the result of a merging), then
    > if accepted, calls output_layers as other layers
    > if rejected, calls rejected_layer (possibly upstream)
    Useful to create "until convergence" loops.
    Encapsulates a CheckingABC module.

    WARNING:
    For now, this layer temporarily remaps connections between
    rejected_layer and its embedding reference. Therefore, it is not
    suited for very complicated structures with multiple, intertwined
    loops.
    """

    def __init__(
        self,
        checking: CheckingABC,
        n_checks_max: int = 10,
    ) -> None:
        Layer.__init__(self, compatible_inputs=[IsRepresentable], str_type="CHECKING")
        IsWatchable.__init__(self, compatible_watchers=[WatcherTiming])
        IsProfilable.__init__(self)
        IsRepresentable.__init__(self, repr_key=f"{self}_{self.layer_id}")
        self.checking = checking
        self.n_checks = 0  # Numbers of checkings done
        self.n_checks_max = n_checks_max  # Max checks allowed
        self.rejected_layer: Optional[Layer] = None
        self.rejected_layer_ref: Optional[IsRepresentable] = None

    def connect_rejected(self, layer: Layer):
        """
        Sets up the rejected connection.
        """
        assert_type(layer, (LayerChecking, LayerMatching, LayerPreprocessing))
        self.rejected_layer = layer
        self.rejected_layer_ref = self.rejected_layer.embedding_reference

    @profile_method
    def fit(self, datasets: List[AnnData]) -> List[Layer]:
        """
        Runs preprocessings if needed, then calls the internal
        checking. If valid, proceeds to output_layers. Otherwise,
        falls back to rejected_layer.
        """
        assert self.rejected_layer is not None, "A rejected layer must be specified."
        Xs = [self.embedding_reference.get(adata) for adata in datasets]
        for adata, X in zip(datasets, Xs):
            self.write(adata, X)
        if isinstance(self.checking, HasMetadata):
            self.checking.retrieve_all_metadata(datasets)
        self.n_checks += 1
        is_valid = self.n_checks >= self.n_checks_max or self.checking.check(Xs)
        if is_valid:
            if self.n_checks >= self.n_checks_max:
                self._log("Maximum number of checks reached.", level=logging.INFO)
            self._log("Checking passed. Continuing.", level=logging.INFO)
            assert self.rejected_layer_ref is not None
            self.rejected_layer.embedding_reference = self.rejected_layer_ref
            return self.output_layers
        else:
            self._log("Checking failed. Continuing.", level=logging.INFO)
            self.rejected_layer.embedding_reference = self
            return [self.rejected_layer]
