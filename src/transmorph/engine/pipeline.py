#!/usr/bin/env python3

import logging
import warnings
import numpy as np

from anndata import AnnData
from transmorph import logger
from transmorph import settings
from transmorph.utils import anndata_manager as adm
from transmorph.utils import AnnDataKeyIdentifiers
from typing import List, Optional

from .layers import (
    LayerInput,
    LayerOutput,
    Layer,
)
from .watchers import Watcher
from .profiler import Profiler


class TransmorphPipeline:
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

    @staticmethod
    def _log(msg: str, level: int = logging.DEBUG) -> None:
        logger.log(level, f"PIPELINE > {msg}")

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
        self._log("Fetching pipeline...")
        self.input_layer = input_layer
        layers_to_visit: List[Layer] = [self.input_layer]
        self.layers = [self.input_layer]
        while len(layers_to_visit) > 0:
            current_layer = layers_to_visit.pop(0)
            current_layer.profiler = self.profiler
            current_layer.set_verbose(self.verbose)
            for watcher in current_layer.watchers:
                self.watchers.append(watcher)
            for output_layer in current_layer.output_layers:
                if output_layer in self.layers:
                    continue
                if type(output_layer) is LayerOutput:
                    self.output_layers.append(output_layer)
                layers_to_visit.append(output_layer)
                self.layers.append(output_layer)
        if len(self.output_layers) == 0:
            warnings.warn(
                "No output layer reachable from input. This pipeline will not "
                "write results in AnnData objects."
            )
        if len(self.output_layers) > 1:  # Temp
            raise NotImplementedError("No more than one output allowed.")
        self._log(
            f"Pipeline initialized -- {len(self.layers)} layers, "
            f"{len(self.output_layers)} outputs, "
            f"{len(self.watchers)} watchers found.",
            level=logging.INFO,
        )

    def _check_input(
        self,
        datasets: List[AnnData],
        reference: Optional[AnnData] = None,
        use_rep: Optional[str] = None,
    ):
        self._log("Checking input...")

        # Check datasets are of the right type, casting them if necessary
        for i, adata in enumerate(datasets):
            if type(adata) is not AnnData:
                assert type(adata) is np.ndarray, (
                    f"Unrecognized dataset type: {type(adata)}. Please provide your "
                    "data as AnnData objects. Numpy arrays are tolerated if no "
                    "metadata is required."
                )
                warnings.warn(
                    "AnnData expected as input, np.ndarray found. Casting to AnnData. "
                    "Gene names being absent can cause inconsistencies in the pipeline."
                )
                datasets[i] = AnnData(adata)

        # Ensure representation is available
        if use_rep is not None:
            for adata in datasets:
                assert use_rep in adata.obsm, f"KeyError: {use_rep}"

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
        self._check_input(datasets, reference, use_rep)
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
            else:
                base_rep = adata.X
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
        self._log(
            f"Ready to start the integration of {ndatasets} datasets,"
            f" {nsamples} total samples.",
            level=logging.INFO,
        )

        # Running
        layers_to_run = [(None, self.input_layer)]
        while len(layers_to_run) > 0:
            caller, called = layers_to_run.pop(0)
            output_layers = called.fit(caller, datasets)
            for watcher in called.watchers:
                watcher.compute(datasets)
            layers_to_run += [(called, out) for out in output_layers]
            for adata in datasets:
                adm.clean(adata, "layer")

        # Cleaning
        for adata in datasets:
            adm.clean(adata, "pipeline")

        # Logging summary
        if len(self.output_layers) > 0:
            npoints = sum(adata.n_obs for adata in datasets)
            ndims = datasets[0].obsm["transmorph"].shape[1]
            self._log(
                f"Terminated. Embedding shape: {(npoints, ndims)}", level=logging.INFO
            )
        self._log(
            "### REPORT_START ###\n"
            + self.profiler.log_stats()
            + "\n"
            + self.profiler.log_tasks()
        )
        self._log("### REPORT_END ###")
