#!/usr/bin/env python3

import logging
import numpy as np

from anndata import AnnData
from typing import List, Optional

from .layers import Layer, LayerChecking, LayerInput, LayerOutput
from .traits import CanLog, CanCatchChecking, IsWatchable, UsesNeighbors
from .. import profiler
from .watchers import Watcher
from .. import settings
from ..utils import anndata_manager as adm, AnnDataKeyIdentifiers


class Model(CanLog):
    """
    Model wraps a layers network in order to represent an integration
    algorithm. It needs at least one LayerInput and one LayerOutput.

    Initialization
    --------------
    Initializing a Model requires a connected network of
    Layers. Network source must be an LayerInput, and finish with one
    or more LayerOutput.

    Running
    -------
    To run the Model, simply use the .run(List[AnnData]) method. It
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
    >>> tp = Model()
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
                assert isinstance(current_layer.rejected_layer, Layer)
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
        assert len(datasets) > 0, "No dataset provided."
        assert self.input_layer is not None, "Pipeline must be initialized first."

        # Check datasets are of the right type, casting them if necessary
        # TODO: raise hard error if obs/var names needed.
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

        # Flags reference dataset if any
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

        # Computes NN graph if needed
        if UsesNeighbors.Used:
            if settings.neighbors_use_scanpy:
                UsesNeighbors.set_settings_to_scanpy(datasets[0])
            UsesNeighbors.compute_neighbors_graphs(
                datasets,
                representation_key=AnnDataKeyIdentifiers.BaseRepresentation,
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
            if isinstance(called, CanCatchChecking) and called.called_by_checking:
                called.restore_previous_mapping()
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
            + profiler.log_stats()
            + "\n"
            + profiler.log_tasks()
        )
        self.log("### REPORT_END ###")
