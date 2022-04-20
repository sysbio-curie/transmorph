#!/usr/bin/env python3

from __future__ import annotations

import logging
import numpy as np

from abc import ABC, abstractmethod
from anndata import AnnData
from typing import List, Optional, Type

from transmorph import settings
from transmorph.engine.checking import LayerChecking
from transmorph.engine.traits import CanLog, IsRepresentable, assert_trait
from transmorph.engine.profiler import Profiler
from transmorph.engine.watchers import IsWatchable, Watcher
from transmorph.utils import anndata_manager as adm
from transmorph.utils import AnnDataKeyIdentifiers
from transmorph.utils.type import assert_type


class Layer(ABC, CanLog):
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
        str_identifier: str = "BASE",
    ) -> None:
        CanLog.__init__(self, str_identifier=f"LAYER_{str_identifier}#{Layer.LayerID}")
        self.layer_id = Layer.LayerID
        Layer.LayerID += 1
        self.compatible_inputs = compatible_inputs
        self.input_layer: Optional[Layer] = None
        self.output_layers: List[Layer] = []
        self.profiler = None
        self._embedding_reference = None
        self.time_elapsed = -1
        self.log("Initialized.")

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
        self.log(f"Connected to layer {layer}.")
        if layer.embedding_reference is None:
            if not isinstance(self, IsRepresentable):
                reference = self.embedding_reference
            else:
                reference = self
            self.log(f"{reference} chosen as default embedding reference for {self}.")
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
                self.raise_error(
                    ValueError,
                    "Input layer is None. Please make sure the "
                    "pipeline contains at least an input layer.",
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
        Layer.__init__(self, compatible_inputs=[], str_identifier="INPUT")
        IsRepresentable.__init__(
            self, repr_key=AnnDataKeyIdentifiers.BaseRepresentation
        )

    def fit(self, datasets: List[AnnData]) -> List[Layer]:
        """
        Simply calls the downstream layers.
        """
        self.log("Checking all representations are present.")
        for adata in datasets:
            assert (
                adm.get_value(adata, self.repr_key) is not None
            ), f"Representation {self.repr_key} missing in {adata}."
        self.log("All representations found. Continuing.")
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
            str_identifier="OUTPUT",
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
            self.write(adata, X, self.embedding_reference.is_feature_space)
        return []


class TransmorphPipeline(CanLog):
    """
    TransmorphPipeline wraps a layers network in order to represent an integration
    pipeline. It needs at least one LayerInput and one LayerOutput.

    Initialization
    --------------
    Initializing a TransmorphPipeline requires a connected network of
    Layers. Network source must be an LayerInput, and finish with one
    or more LayerOutput.

    Running
    -------
    To run the TransmorphPipeline, simply use the .run(List[AnnData]) method. It
    will then recursively apply layers.

    Output
    ------
    For simplicity, transmorph directly writes integration result in the AnnData
    object under the entry .obsm['X_transmorph_$i'] where $i is the output layer id.

    Attributes
    ----------
    - layer_input: LayerInput, entry node in the network.
    - output_layers: List[LayerOutput], output nodes
    - layers: List[Layer], set of layers connected to self.layer_input

    Example
    -------
    >>> li = LayerInput()
    >>> l1, l2, ... # Various computational layers
    >>> lo = LayerOutput()
    >>> li.connect(l1)
    >>> ...
    >>> ln.connect(lo)
    >>> tp = TransmorphPipeline()
    >>> tp.initialize(li)
    >>> tp.run([adata1, adata2, adata3])
    >>> adata1.obsm['X_transmorph_0']
    -> np.ndarray(shape=(n,d))
    """

    def __init__(self, verbose: bool = False) -> None:
        CanLog.__init__(self, str_identifier="PIPELINE")
        self.input_layer = None
        self.output_layers = []
        self.layers: List[Layer] = []
        self.watchers: List[Watcher] = []
        self.profiler = Profiler()
        self.verbose = verbose
        if verbose:
            settings.verbose = "INFO"
        else:
            settings.verbose = "WARNING"

    def initialize(self, input_layer: LayerInput):
        """
        Loads the network and checks basic layout properties.

        Parameters
        ----------
        input_layer: LayerInput
            Entry point of the pipeline. Must have been conneced to the other layers
            beforehand.
        """
        assert type(input_layer) is LayerInput, "LayerInput expected."
        self.log("Fetching pipeline...")
        self.input_layer = input_layer
        layers_to_visit: List[Layer] = [self.input_layer]
        self.layers = [self.input_layer]
        while len(layers_to_visit) > 0:
            current_layer = layers_to_visit.pop(0)
            if (
                isinstance(current_layer, LayerChecking)
                and current_layer.rejected_layer is not None
            ):
                output_layers = current_layer.output_layers + [
                    current_layer.rejected_layer
                ]
            else:
                output_layers = current_layer.output_layers
            for output_layer in output_layers:
                if output_layer in self.layers:
                    continue
                if type(output_layer) is LayerOutput:
                    self.output_layers.append(output_layer)
                layers_to_visit.append(output_layer)
                self.layers.append(output_layer)
        if len(self.output_layers) == 0:
            self.warn(
                "No output layer reachable from input. This pipeline will not "
                "write results in AnnData objects."
            )
        if len(self.output_layers) > 1:  # Temp
            raise NotImplementedError("No more than one output allowed.")
        self.log(
            f"Pipeline initialized -- {len(self.layers)} layers, "
            f"{len(self.output_layers)} outputs, "
            f"{len(self.watchers)} watchers found.",
            level=logging.INFO,
        )

    def fit(
        self,
        datasets: List[AnnData],
        reference: Optional[AnnData] = None,
        use_rep: Optional[str] = None,
    ):
        """
        Runs the pipeline given a list of AnnDatas, writes integration results in
        each AnnData object under the entry .obsm['X_transmorph_$i'] where $i is
        the output layer id.

        Parameters
        ----------
        datasets: List[AnnData]
            Datasets to integrate.

        reference: AnnData
            Reference AnnData to use in mergings that need one. If the pipeline
            is reference-less, leave it None.

        use_representation: str
            Matrix representation to use during the pipeline, useful to provide
            a low-dimensional representation of data.
            If None, use AnnData.X. Otherwise, use AnnData.obsm[use_rep].

        Usage
        -----
        >>> tp.run([adata1, adata2, adata3])
        >>> adata1.obsm['X_transmorph_0']
        -> np.ndarray(shape=(n,d))
        """
        # Initializing
        assert self.input_layer is not None, "Pipeline must be initialized first."

        # Check datasets are of the right type, casting them if necessary
        for i, adata in enumerate(datasets):
            if type(adata) is not AnnData:
                assert type(adata) is np.ndarray, (
                    f"Unrecognized dataset type: {type(adata)}. Please provide your "
                    "data as AnnData objects. Numpy arrays are tolerated if no "
                    "metadata is required."
                )
                self.warn(
                    "AnnData expected as input, np.ndarray found. Casting to AnnData. "
                    "Gene names being absent can cause inconsistencies in the pipeline."
                )
                datasets[i] = AnnData(adata)

        # Sets reference if necessary
        if reference is not None:
            for adata in datasets:
                adm.set_value(
                    adata=adata,
                    key=AnnDataKeyIdentifiers.IsReference,
                    field="uns",
                    value=adata is reference,
                    persist="pipeline",
                )

        # Setting base representation
        for adata in datasets:
            if use_rep is not None:
                base_rep = adata.obsm[use_rep]
                self.input_layer.is_feature_space = False
            else:
                base_rep = adata.X
                self.input_layer.is_feature_space = True
            adm.set_value(
                adata=adata,
                key=AnnDataKeyIdentifiers.BaseRepresentation,
                field="obsm",
                value=base_rep,
                persist="pipeline",
            )

        # Logging some info
        ndatasets = len(datasets)
        nsamples = sum([adata.n_obs for adata in datasets])
        self.log(
            f"Ready to start the integration of {ndatasets} datasets,"
            f" {nsamples} total samples.",
            level=logging.INFO,
        )

        # Running
        layers_to_run = [self.input_layer]
        while len(layers_to_run) > 0:
            called = layers_to_run.pop(0)
            output_layers = called.fit(datasets)
            if isinstance(called, IsWatchable):
                called.update_watchers()
            layers_to_run += [output_layers]
            for adata in datasets:
                adm.clean(adata, "layer")

        # Cleaning
        for adata in datasets:
            adm.clean(adata, "pipeline")

        # Logging summary
        if len(self.output_layers) > 0:
            npoints = sum(adata.n_obs for adata in datasets)
            ndims = datasets[0].obsm["transmorph"].shape[1]
            self.log(
                f"Terminated. Embedding shape: {(npoints, ndims)}", level=logging.INFO
            )
        self.log(
            "### REPORT_START ###\n"
            + self.profiler.log_stats()
            + "\n"
            + self.profiler.log_tasks()
        )
        self.log("### REPORT_END ###")
