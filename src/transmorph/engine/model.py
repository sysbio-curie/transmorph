#!/usr/bin/env python3

from anndata import AnnData
from typing import List, Literal, Optional

from .layers import Layer, LayerChecking, LayerInput, LayerOutput
from .traits import CanLog, CanCatchChecking, IsProfilable, UsesNeighbors, UsesReference
from .. import profiler
from ..utils import anndata_manager as adm, AnnDataKeyIdentifiers


class Model(CanLog):
    """
    A Model wraps and manages a network of Layers, allowing to articulate
    integration modules together. Layers are expected to be already
    connected to one another, with exactly one input and one output layer.

    Parameters
    ----------
    input_layer: LayerInput
        Input layer of the network, which is assumed to be already
        connected. The Model will then automatically gather all
        layers connected to the network, which is expected to have
        exactly one output layer.

    verbose: Literal["DEBUG", "INFO", "WARNING", "ERROR"], default = "INFO"
        Level of verbose of console logging.

        - "DEBUG": All loggings are printed to the console
        - "INFO": Only important informations are printed (default)
        - "WARNING": Only warnings and errors are printed
        - "ERROR": No information is printed, except for runtime errors
    """

    def __init__(
        self,
        input_layer: LayerInput,
        verbose: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    ) -> None:
        from .. import settings

        CanLog.__init__(self, str_identifier="PIPELINE")
        self.output_layers = []
        self.layers: List[Layer] = []
        assert isinstance(
            input_layer, LayerInput
        ), f"LayerInput expected, {type(input_layer)} found."
        self.input_layer = input_layer
        self._initialize()
        settings.verbose = verbose

    def _initialize(self):
        """
        Loads the network and checks basic layout properties.
        Is called by __init__.
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
        Runs the Model given a list of AnnData datasets, and writes integration
        results in each AnnData object under the entry .obsm['transmorph']. A
        reference dataset, or a specific dataset .obsm representation can be
        set if needed.

        Parameters
        ----------
        datasets: List[AnnData]
            Datasets to integrate at the AnnData format. Basic preprocessing such
            as sample filtering is expected at this stage, even if additional
            transformation steps can be integrated in a Model. If you expect
            the Model to work with datasets in a common feature space, please
            ensure var_names are common between datasets -- not necessarily
            all identical and in the same order, but at least with common names.

        reference: AnnData
            If some parts of the pipeline (typically mergings) are expected
            to work with a reference, you must provide the reference AnnData
            using this parameter (it must also be part of datasets list).
            TODO: Automatically detect at initialization if reference is needed,
            to avoid useless computations before raising an error.

        use_representation: str
            Matrix representation to use during the pipeline, useful to provide
            a low-dimensional representation of data.
            If None, use AnnData.X. Otherwise, uses the provided .obsm key.

        Returns
        -------
        fit() will write the datasets at entry .obsm["transmorph"] with
        integrated embeddings.
        """
        # Sanity checks
        assert len(datasets) > 0, "No dataset provided."
        assert self.input_layer is not None, "Pipeline must be initialized first."
        assert len(self.output_layers) == 1, "No output layer found."
        assert all(isinstance(adata, AnnData) for adata in datasets), (
            "Only AnnData objects can be processed by a Model. "
            "You can create one from a numpy ndarray using "
            "AnnData(X: np.ndarray). Note that in this case, "
            "observations and variables metadata will be attributed "
            "automatically, and may cause incompatibilities for some "
            "pipeline steps."
        )

        self.info("Transmorph model is initializing.")

        # Flags reference dataset as such if any
        for adata in datasets:
            self.log("Flagging reference dataset.")
            if adata is reference:
                UsesReference.write_is_reference(adata)

        # Setting base representation if needed
        if use_representation is not None:
            self.log("Setting base representation.")
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
            self.log("Precomputing neighbors graph.")
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
            self.info(f"Running layer {called}.")
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

        # Cleaning anndatas, LayerOutput has saved the result
        adm.clean(datasets, "pipeline")

        # Logging summary

        loutput = self.output_layers[0]
        npoints = sum(adata.n_obs for adata in datasets)
        ndims = loutput.get_representation(datasets[0]).shape[1]
        self.info(f"Terminated. Total embedding shape: {(npoints, ndims)}")
        self.log(
            "### REPORT_START ###\n"
            + profiler.log_stats()
            + "\n"
            + profiler.log_tasks()
        )
        self.log("### REPORT_END ###")
