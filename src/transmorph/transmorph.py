#!/usr/bin/env python3

from abc import abstractmethod
from enum import Enum, auto
from typing import List, Union

from scipy.sparse.csr import csr_matrix

from .checking.checkingABC import CheckingABC
from .matching.matchingABC import MatchingABC
from .merging.mergingABC import MergingABC

from anndata import AnnData
from .utils.anndata_interface import (
    delete_attribute,
    delete_matrix,
    get_attribute,
    get_matrix,
    set_attribute,
    set_matrix,
)


class LayerType(Enum):
    """
    Specifies layer types for easy type checking.
    """

    BASE = auto()
    INPUT = auto()
    OUTPUT = auto()
    MATCH = auto()
    MERGE = auto()
    CHECK = auto()


class LayerTransmorph:
    """
    A LayerTransmorph wraps an integration module, and manage its connections
    with other modules. All Layers derive from this class.

    Parameters
    ----------
    type: LayerType, default = LayerType.BASE
        String identifier describing module type. Can be used for compatibility
        purpose.

    compatible_types: List[LayerType]
        List of type identifiers of compatible input layers

    maximum_inputs: int = -1
        Maximal number of inputs, set it to -1 if there is no maximum

    Attributes
    ----------
    output_layers: List[LayerTransmorph]
        List of outgoing target layers.

    TODO: cacheing
    """

    LayerID = 0

    def __init__(
        self,
        layer_type: LayerType = LayerType.BASE,
        compatible_inputs: List[LayerType] = [],
        verbose: bool = False,
    ) -> None:
        self.type = layer_type
        self.compatible_inputs = compatible_inputs
        self.output_layers: List["LayerTransmorph"] = []
        self.verbose = verbose
        self.LayerID = LayerTransmorph.LayerID
        LayerTransmorph.LayerID += 1
        self._log("Initialized.")

    def _log(self, msg: str):
        if not self.verbose:
            return
        print(f"{self} >", msg)

    def __str__(self):
        return f"({self.type} {self.LayerID})"

    def connect(self, layer: "LayerTransmorph") -> None:
        """
        Connects the current layer to an output layer, if compatible.

        Parameters
        ----------
        layer: LayerTransmorph
            Output layer of compatible type.
        """
        assert (
            self.type in layer.compatible_inputs
        ), f"Incompatible connection: {self} -> {layer}"
        assert layer not in self.output_layers, "{self} already connected to {layer}."
        self.output_layers.append(layer)
        self._log(f"Connected to layer {layer}.")

    @abstractmethod
    def fit(self, caller: "LayerTransmorph", datasets: List[AnnData]):
        """
        This is the computational method, running an internal module.
        It then should write its output in the AnnDatas, and callback
        the downstream fit() methods.

        Parameters
        ----------
        caller: LayerTransmorph
            Reference to current layer, used to retrieve relevant information
            relative to computation results.

        datasets: List[AnnData]
            List of datasets to process.
        """
        pass

    def clean(self, datasets: List[AnnData]):
        """
        TODO
        """
        raise NotImplementedError

    def get_representation(self) -> str:
        """
        TODO
        """
        raise NotImplementedError

    def get_matching(self, adata_src: AnnData, adata_ref: AnnData) -> csr_matrix:
        """
        TODO
        """
        raise NotImplementedError


class LayerInput(LayerTransmorph):
    """
    Every pipeline must contain exactly one input layer. Every pipeline
    is initialized using this layer.
    """

    def __init__(self, verbose: bool = False) -> None:
        super().__init__(
            layer_type=LayerType.INPUT, compatible_inputs=[], verbose=verbose
        )

    def fit(self, caller: LayerTransmorph, datasets: List[AnnData]):
        """
        Simply calls the downstream layers.
        """
        assert caller is None, f"{caller} called {self}."
        self._log("Calling next layers.")
        for output in self.output_layers:
            output.fit(self, datasets)

    def clean(self, datasets: List[AnnData]):
        self._log("Cleaning.")
        for output in self.output_layers:
            output.clean(datasets)

    def get_representation(self) -> str:
        """
        Returns a matrix representation of AnnData.
        """
        return ""


class LayerOutput(LayerTransmorph):
    """
    Simple layer to manage network outputs. There can be several output layers.
    TODO
    """

    def __init__(self, verbose: bool = False) -> None:
        super().__init__(
            layer_type=LayerType.OUTPUT,
            compatible_inputs=[LayerType.CHECK, LayerType.INPUT, LayerType.MERGE],
            verbose=verbose,
        )
        self.representation_kw = ""

    def fit(self, caller: LayerTransmorph, datasets: List[AnnData]):
        """
        Runs the upstream pipeline and stores results in AnnData objects.
        """
        self._log("Retrieving keyword.")
        self.representation_kw = caller.get_representation()
        self._log(f"Found '{self.representation_kw}'. Terminating the branch.")

    def clean(self, datasets: List[AnnData]):
        self._log("Cleaning ended for this branch.")

    def get_representation(self) -> str:
        return self.representation_kw


class LayerMatching(LayerTransmorph):
    """
    This layer performs a matching between two or more datasets.
    It wraps an object derived from MatchingABC.
    """

    def __init__(self, matching: MatchingABC, verbose: bool = False) -> None:
        super().__init__(
            layer_type=LayerType.MATCH,
            compatible_inputs=[
                LayerType.CHECK,
                LayerType.INPUT,
                LayerType.MERGE,
                LayerType.CHECK,
            ],
            verbose=verbose,
        )
        self.matching = matching
        self.fitted = False
        self.representation_kw = ""

    def fit(self, caller: LayerTransmorph, datasets: List[AnnData]):
        """
        TODO
        """
        self._log("Requesting keyword.")
        self.representation_kw = caller.get_representation()
        self._log(f"Found '{self.representation_kw}'. Calling matching.")
        self.matching.fit(datasets, self.representation_kw)
        self.fitted = True
        self._log("Fitted.")
        for layer in self.output_layers:
            layer.fit(self, datasets)

    def clean(self, datasets: List[AnnData]):
        self._log("Cleaning.")
        for output in self.output_layers:
            output.clean(datasets)

    def get_representation(self) -> str:
        """
        TODO
        """
        assert self.fitted, "{self} must be fitted to access its representation."
        return self.representation_kw

    def get_matching(self, adata_src: AnnData, adata_ref: AnnData) -> csr_matrix:
        """
        TODO
        """
        assert self.fitted, "{self} must be fitted to access its matching."
        return self.matching.get_matching(adata_src, adata_ref)


class LayerMerging(LayerTransmorph):
    """
    TODO
    """

    def __init__(self, merging: MergingABC, verbose: bool = False) -> None:
        """
        TODO
        """
        super().__init__(
            layer_type=LayerType.MERGE,
            compatible_inputs=[LayerType.CHECK, LayerType.MATCH],  # TODO test check
            verbose=verbose,
        )
        self.merging = merging
        self.use_reference = merging.use_reference
        self.fitted = False
        self.mtx_id = f"merging_{self.LayerID}"

    def fit(self, caller: LayerMatching, datasets: List[AnnData]):
        """
        TODO
        """
        self._log("Requesting keyword.")
        representation_kw = caller.get_representation()
        self._log(f"Found '{representation_kw}'. Calling merging.")
        ref_id = -1
        if self.use_reference:
            for k, adata in enumerate(datasets):
                if get_attribute(adata, "is_reference"):
                    ref_id = k
                    break
            assert (
                ref_id != -1
            ), "Error: No reference found in TransmorphPipeline.fit()."
        X_transform = self.merging.fit(
            datasets,
            matching=caller.matching,
            X_kw=representation_kw,
            reference_idx=ref_id,
        )
        for adata, X_after in zip(datasets, X_transform):
            set_matrix(adata, self.mtx_id, X_after)
        self.fitted = True
        self._log("Fitted.")
        for layer in self.output_layers:
            layer.fit(self, datasets)

    def clean(self, datasets: List[AnnData]):
        self._log("Cleaning.")
        for adata in datasets:
            delete_matrix(adata, self.mtx_id)
        for output in self.output_layers:
            output.clean(datasets)

    def get_representation(self) -> str:
        return self.mtx_id


class LayerChecking(LayerTransmorph):
    """
    TODO
    """

    def __init__(
        self, checking: CheckingABC, n_checks_max: int = 10, verbose: bool = False
    ) -> None:
        super().__init__(
            layer_type=LayerType.CHECK,
            compatible_inputs=[LayerType.CHECK, LayerType.MERGE],  # Test CHECK
            verbose=verbose,
        )
        self.checking = checking
        self.n_checks = 0
        self.n_checks_max = n_checks_max
        self.layer_yes: Union[None, LayerTransmorph] = None
        self.layer_no: Union[None, LayerTransmorph] = None
        self.mtx_id = f"checking_{self.LayerID}"
        self.cleaned = False  # TODO improve this to prevent critical bugs

    def connect(self, layer: LayerTransmorph):
        raise NotImplementedError(
            "Please use instead connect_yes and connect_no for LayerChecking."
        )

    def connect_yes(self, layer: LayerTransmorph):
        assert self.layer_yes is None, "Error: Only one layer 'YES' is allowed."
        super().connect(layer)
        self.layer_yes = layer

    def connect_no(self, layer: LayerTransmorph):
        assert self.layer_no is None, "Error: Only one layer 'NO' is allowed."
        super().connect(layer)
        self.layer_no = layer

    def fit(self, caller: LayerTransmorph, datasets: List[AnnData]):
        """
        TODO
        """
        assert self.layer_yes is not None, "Error: No layer found for 'YES' path."
        assert self.layer_no is not None, "Error: No layer found for 'NO' path."
        self._log("Requesting keyword.")
        representation_kw = caller.get_representation()
        self._log(f"Found '{representation_kw}'. Checking validity.")
        for adata in datasets:
            set_matrix(adata, self.mtx_id, get_matrix(adata, representation_kw))
        valid = self.checking.check(datasets, self.mtx_id)
        self.n_checks += 1
        if valid or self.n_checks >= self.n_checks_max:
            if not valid:
                self._log("Warning, number of checks exceeded, validating by default.")
            self._log("Checking loop ended, pursuing.")
            self.layer_yes.fit(self, datasets)
        else:
            self._log("Check fail, retrying.")
            self.layer_no.fit(self, datasets)

    def clean(self, datasets: List[AnnData]):
        if self.cleaned:
            return
        self.cleaned = True
        self._log("Cleaning.")
        for adata in datasets:
            delete_matrix(adata, self.mtx_id)
        for output in self.output_layers:
            output.clean(datasets)

    def get_representation(self) -> str:
        return self.mtx_id


# class LayerCombineMatching(LayerTransmorph):
#     """
#     TODO
#     """

#     def __init__(self, mode: str) -> None:
#         raise NotImplementedError

#     def get_datasets(self):
#         """
#         TODO
#         """
#         raise NotImplementedError

#     def normalize(self, T_matching):
#         """
#         TODO
#         """
#         raise NotImplementedError

#     def run(self, datasets):
#         """
#         TODO
#         """
#         raise NotImplementedError


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
            for output_layer in current_layer.output_layers:
                if output_layer in self.layers:
                    continue
                if output_layer.type == LayerType.OUTPUT:
                    self.output_layers.append(output_layer)
                layers_to_visit.append(output_layer)
                self.layers.append(output_layer)
        assert len(self.output_layers) > 0, "No output layer reachable from input."
        self._log(
            f"Terminated -- {len(self.layers)} layers, "
            f"{len(self.output_layers)} output found."
        )

    def fit(self, datasets: List[AnnData], reference: AnnData = None):
        """
        Runs the pipeline given a list of AnnDatas, writes integration results in
        each AnnData object under the entry .obsm['X_transmorph_$i'] where $i is
        the output layer id.

        Parameters
        ----------
        datasets: List[AnnData]
            Datasets to integrate.

        Usage
        -----
        >>> tp.run([adata1, adata2, adata3])
        >>> adata1.obsm['X_transmorph_0']
        -> np.ndarray(shape=(n,d))
        """
        assert self.input_layer is not None, "Pipeline must be initialized first."
        if reference is not None:
            for adata in datasets:
                if adata is not reference:
                    set_attribute(adata, "is_reference", False)
                else:
                    set_attribute(adata, "is_reference", True)
        self.input_layer.fit(None, datasets)
        output_kw = self.output_layers[
            0
        ].get_representation()  # TODO several output layers
        for adata in datasets:
            adata.obsm["transmorph"] = get_matrix(adata, output_kw)
        self.input_layer.clean(datasets)
        if reference is not None:
            for adata in datasets:
                delete_attribute(adata, "is_reference")
