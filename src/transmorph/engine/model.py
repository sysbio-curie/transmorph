#!/usr/bin/env python3

from anndata import AnnData
from typing import List, Optional

from transmorph.engine.traits.isprofilable import IsProfilable

from .layers import Layer, LayerChecking, LayerInput, LayerOutput
from .traits import CanLog, CanCatchChecking, UsesNeighbors, UsesReference
from .. import profiler
from ..utils import anndata_manager as adm, AnnDataKeyIdentifiers


class Model(CanLog):
    """
    TODO update description
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
    """

    def __init__(self, input_layer: LayerInput, verbose: bool = True) -> None:
        from .. import settings

        CanLog.__init__(self, str_identifier="PIPELINE")
        self.output_layers = []
        self.layers: List[Layer] = []
        assert isinstance(
            input_layer, LayerInput
        ), f"LayerInput expected, {type(input_layer)} found."
        self.input_layer = input_layer
        self.initialize()
        if verbose:
            settings.verbose = "INFO"
        else:
            settings.verbose = "WARNING"

    def initialize(self):
        """
        Loads the network and checks basic layout properties.

        Parameters
        ----------
        input_layer: LayerInput
            Entry point of the pipeline. Must have been conneced to the other layers
            beforehand.
        """
        self.log("Fetching pipeline layers...")
        layers_to_visit: List[Layer] = [self.input_layer]
        self.layers = [self.input_layer]
        while len(layers_to_visit) > 0:
            current_layer = layers_to_visit.pop(0)
            self.log(f"Branching from layer {current_layer}.")

            # Retrieving output layers
            output_layers = current_layer.output_layers.copy()
            # Checking layers have an extra output
            if (
                isinstance(current_layer, LayerChecking)
                and current_layer.rejected_layer is not None
            ):
                assert isinstance(current_layer.rejected_layer, Layer)
                output_layers.append(current_layer.rejected_layer)

            # Registering non-visited output layers
            for output_layer in output_layers:
                if output_layer in self.layers:
                    self.log(f"Ignored connection to {output_layer}.")
                    continue
                if isinstance(output_layer, LayerOutput):
                    self.output_layers.append(output_layer)
                self.log(f"Found connection to {output_layer}.")
                layers_to_visit.append(output_layer)
                self.layers.append(output_layer)

        # Verifying structure is correct
        # FIXME: Analyze checking loops
        if len(self.output_layers) == 0:
            self.warn(
                "No output layer reachable from input. This pipeline will not "
                "write results in AnnData objects."
            )
        # TODO: Could we allow this?
        noutputs = len(self.output_layers)
        assert noutputs == 1, f"Exactly one output allowed, found {noutputs}."
        self.log(
            f"Pipeline initialized -- {len(self.layers)} layers found, "
            f"{len(self.output_layers)} outputs found.",
        )

    def fit(
        self,
        datasets: List[AnnData],
        reference: Optional[AnnData] = None,
        use_representation: Optional[str] = None,
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
        """
        # Sanity checks
        assert len(datasets) > 0, "No dataset provided."
        assert self.input_layer is not None, "Pipeline must be initialized first."
        assert all(isinstance(adata, AnnData) for adata in datasets), (
            "Only AnnData objects can be processed by a Model. "
            "You can create one from a numpy ndarray using "
            "AnnData(X: np.ndarray). Note that in this case, "
            "observations and variables metadata will be attributed "
            "automatically, and may cause incompatibilities for some "
            "pipeline steps."
        )

        # Flags reference dataset if any
        for adata in datasets:
            if adata is reference:
                UsesReference.write_is_reference(adata)

        # Setting base representation if needed
        if use_representation is not None:
            for adata in datasets:
                adm.set_value(
                    adata=adata,
                    key=AnnDataKeyIdentifiers.BaseRepresentation,
                    field="obsm",
                    value=adata.obsm[use_representation],
                    persist="pipeline",
                )

        # Computes NN graph if needed
        if UsesNeighbors.Used:
            self.info("Precomputing neighbors graph.")
            UsesNeighbors.compute_neighbors_graphs(
                datasets,
                representation_key=AnnDataKeyIdentifiers.BaseRepresentation,
            )

        # Logging some info
        ndatasets = len(datasets)
        nsamples = sum([adata.n_obs for adata in datasets])
        self.info(
            f"Ready to start the integration of {ndatasets} datasets,"
            f" {nsamples} total samples.",
        )

        # Running
        layers_to_run = [self.input_layer]
        while len(layers_to_run) > 0:
            called = layers_to_run.pop(0)
            output_layers = called.fit(datasets)
            # Invalid checking -> layer rejected
            if isinstance(called, LayerChecking) and called.check_is_valid:
                assert len(output_layers) == 1
                output_layer = output_layers[0]
                assert isinstance(output_layer, CanCatchChecking)
                output_layer.called_by_checking = True
            # Rejected layer was computed, can fall back to
            # previous representation
            if isinstance(called, CanCatchChecking) and called.called_by_checking:
                called.called_by_checking = False
            layers_to_run += output_layers
            adm.clean(datasets, "layer")
            if isinstance(called, IsProfilable):
                self.log(f"Layer {called} terminated in {called.get_time_spent()}")

        # Cleaning, LayerOutput has saved the result
        adm.clean(datasets, "pipeline")

        # Logging summary
        loutput = self.output_layers[0]
        if len(self.output_layers) > 0:
            npoints = sum(adata.n_obs for adata in datasets)
            ndims = loutput.get_representation(datasets[0]).shape[1]
            self.info(f"Terminated. Embedding shape: {(npoints, ndims)}")
        self.log(
            "### REPORT_START ###\n"
            + profiler.log_stats()
            + "\n"
            + profiler.log_tasks()
        )
        self.log("### REPORT_END ###")
