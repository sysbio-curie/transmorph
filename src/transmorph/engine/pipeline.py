#!/usr/bin/env python3

from anndata import AnnData
from typing import List, Union

from .layers import LayerType, LayerInput, LayerTransmorph
from .profiler import Profiler
from ..utils.anndata_interface import (
    set_attribute,
    delete_attribute,
    get_matrix,
    set_matrix,
    delete_matrix,
)


class TransmorphPipeline:
    """
    TransmorphPipeline wraps a layers network in order to represent an integration
    pipeline. It needs at least one LayerInput and one LayerOutput.

    Initialization
    --------------
    Initializing a TransmorphPipeline requires a connected network of
    LayerTransmorphs. Network source must be an LayerInput, and finish with one
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
    - layers: List[LayerTransmorph], set of layers connected to self.layer_input

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
        self.layers = []
        self.verbose = verbose
        self.profiler = Profiler()

    def _log(self, msg: str):
        if not self.verbose:
            return
        print(f"{self} >", msg)

    def __str__(self):
        return "(TransmorphPipeline)"

    def initialize(self, input_layer: LayerInput):
        """
        Loads the network and checks basic layout properties.

        Parameters
        ----------
        input_layer: LayerInput
            Entry point of the pipeline. Must have been conneced to the other layers
            beforehand.
        """
        assert input_layer.type == LayerType.INPUT, "LayerInput expected."
        self._log("Fetching pipeline...")
        self.input_layer = input_layer
        layers_to_visit: List[LayerTransmorph] = [self.input_layer]
        self.layers = [layers_to_visit]
        while len(layers_to_visit) > 0:
            current_layer = layers_to_visit.pop(0)
            current_layer.profiler = self.profiler
            for output_layer in current_layer.output_layers:
                if output_layer in self.layers:
                    continue
                if output_layer.type == LayerType.OUTPUT:
                    self.output_layers.append(output_layer)
                layers_to_visit.append(output_layer)
                self.layers.append(output_layer)
        assert len(self.output_layers) > 0, "No output layer reachable from input."
        if len(self.output_layers) > 1:  # Temp
            raise NotImplementedError("No more than one output allowed.")
        self._log(
            f"Terminated -- {len(self.layers)} layers, "
            f"{len(self.output_layers)} output found."
        )

    def fit(
        self,
        datasets: List[AnnData],
        reference: Union[None, AnnData] = None,
        use_rep: Union[None, str] = None,
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
        if reference is not None:
            for adata in datasets:
                if adata is not reference:
                    set_attribute(adata, "is_reference", False)
                else:
                    set_attribute(adata, "is_reference", True)
        if use_rep is not None:
            self.input_layer.use_rep = use_rep
            for adata in datasets:
                set_matrix(adata, use_rep, adata.obsm[use_rep])

        # Running
        layers_to_run = [(None, self.input_layer)]
        while len(layers_to_run) > 0:
            caller, called = layers_to_run.pop(0)
            output_layers = called.fit(caller, datasets)
            layers_to_run += [(called, out) for out in output_layers]

        # TODO several output layers
        output_kw = self.output_layers[0].get_representation()
        for adata in datasets:
            adata.obsm["transmorph"] = get_matrix(adata, output_kw)

        # Cleaning
        self.input_layer.clean(datasets)
        if reference is not None:
            for adata in datasets:
                delete_attribute(adata, "is_reference")
        if use_rep is not None:
            for adata in datasets:
                delete_matrix(adata, use_rep)

        self._log(
            "### REPORT_START ###\n"
            + self.profiler.log_stats()
            + "\n"
            + self.profiler.log_tasks()
        )
        self._log("### REPORT_END ###")
