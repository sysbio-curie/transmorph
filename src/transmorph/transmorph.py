#!/usr/bin/env python3


from abc import abstractmethod
from anndata import AnnData
from typing import Iterator, List

from scipy.sparse.csr import csr_matrix

from transmorph.matching.matchingABC import MatchingABC
from transmorph.matching import MatchingCombined


class LayerTransmorph:
    """
    A LayerTransmorph wraps an integration module, and manage its connections
    with other modules. All Layers derive from this class.

    Parameters
    ----------
    type: str, default = "base"
        String identifier describing module type. Can be used for compatibility
        purpose.

    compatible_types: List[str]
        List of type identifiers of compatible input layers

    maximum_inputs: int = -1
        Maximal number of inputs, set it to -1 if there is no maximum

    remove_output: bool
        Whether layer output can be removed from AnnData when pipeline is finished.

    Attributes
    ----------
    input_layers: List[LayerTransmorph]
        List of incoming source layers.

    output_layers: List[LayerTransmorph]
        List of outgoing target layers.

    output_key: str
        Dictionary key indicating result storage location in AnnData.uns["transmorph"]

    computed: bool
        Indicates whether the results have been computed for cacheing purposes.
    """

    def __init__(
        self,
        type_: str = "base",
        compatible_types: List[str] = [],
        maximum_inputs: int = -1,
        remove_output: bool = True,
    ) -> None:
        self.type: str = type_
        self.compatible_types: List[str] = compatible_types
        self.maximum_inputs: int = maximum_inputs
        self.remove_output: bool = remove_output
        self.input_layers: List["LayerTransmorph"] = []
        self.output_layers: List["LayerTransmorph"] = []
        self.output_data: str = ""
        self.computed: bool = False  # Caches computation results

    def connect(self, layer: "LayerTransmorph") -> None:
        """
        Connects the current layer to an input layer, if compatible and if the
        maximum number of inputs is not reached.

        Parameters
        ----------
        layer: LayerTransmorph
            Input layer of compatible type.
        """
        self.check_connection(layer)
        self.output_layers.append(layer)
        layer.input_layers.append(self)

    def check_connection(self, layer: "LayerTransmorph") -> None:
        """
        Verifies if a candidate layer is valid.
        """
        assert (
            layer.type == "all" or layer.type in self.compatible_types
        ), f"Error: Uncompatible type as input for layer {self.type}: {type(layer)}"
        assert len(self.input_layers) < self.maximum_inputs, (
            f"Error: too many input layers for layer {self.type}. "
            f"Maximum: {self.maximum_inputs}"
        )

    def iter_inputs(self) -> Iterator["LayerTransmorph"]:
        """
        Returns an iterator of layer inputs.
        """
        for layer in self.input_layers:
            yield layer

    def iter_outputs(self) -> Iterator["LayerTransmorph"]:
        """
        Returns an iterator of layer outputs.
        """
        for layer in self.output_layers:
            yield layer

    def set_computed(self, b: bool) -> None:
        """
        For cache purposes. If a layer is "computed", then its function
        .run() immediately returns, until another upstream node changes
        its state.
        """
        # If output changes, propagates the signal downstream
        if b is False:
            for layer in self.output_layers:
                layer.set_computed(b)
        self.computed = b

    @abstractmethod
    def run(self, datasets: List[AnnData]):
        """
        This is the computational method, running an internal module.
        It then should write its output in the AnnDatas, and callback
        the downstream run() methods.
        """
        pass


class LayerInput(LayerTransmorph):
    """
    Every pipeline must contain exactly one input layer. Every pipeline
    is initialized using this layer.
    """

    def __init__(self) -> None:
        super().__init__(
            type_="input",
            compatible_types=["matching", "preprocessing"],
            maximum_inputs=0,
        )

    def run(self, datasets: List[AnnData]):
        """
        Simply calls the downstream layers.
        """
        self.set_computed(False)
        for layer in self.iter_outputs():
            layer.run(datasets)
        self.set_computed(True)


class LayerMatching(LayerTransmorph):
    """
    This layer performs a matching between two or more datasets.
    It wraps an object derived from MatchingABC.
    """

    def __init__(self, matching: MatchingABC) -> None:
        super().__init__(
            "matching",
            ["input", "merging", "preprocessing"],
            maximum_inputs=1,
        )
        self.matching = matching

    def run(self, datasets: List[AnnData]):
        """
        TODO
        """
        # TODO: what if input is pp? merging?
        if not self.computed:
            self.matching.fit(datasets)  # TODO: how to pass the information?
            self.set_computed(True)
        for layer in self.iter_outputs():
            layer.run(datasets)


class LayerCombineMatching(LayerTransmorph):
    """
    TODO
    """

    def __init__(self, mode: str) -> None:
        super().__init__(
            "combine_matching",
            [LayerMerging, LayerCombineMatching],
            1,
        )
        assert mode in ["additive", "multiplicative", "intersection"], (
            f"Unknown mode {mode}. Expected 'additive', 'multiplicative' or"
            "'intersection'."
        )
        self.datasets = []
        self.mode = mode
        self.matching = None

    def get_datasets(self):
        """
        TODO
        """
        return self.datasets

    def normalize(self, T_matching):
        """
        TODO
        """
        return csr_matrix(T_matching / T_matching.sum(axis=1))

    def run(self, datasets):
        """
        TODO
        """
        LayerTransmorph.run(self, datasets)
        assert all(
            type(layer) is LayerMatching or type(layer) is LayerCombineMatching
            for layer in self.input_layers
        ), "At least one input layer is not of Matching type."
        self.matching = MatchingCombined(
            [layer.matching for layer in self.input_layers], mode=self.mode
        )
        # TODO: format the output properly
        self.output_data = self.normalize(self.output_data)


class LayerMerging(LayerTransmorph):
    """
    TODO
    """

    def __init__(self, merging, datasets) -> None:
        """
        TODO
        """
        super().__init__("merging", [LayerMatching, LayerChecking, LayerOutput])
        self.datasets = datasets
        self.merging = merging

    def run(self, datasets):
        """
        TODO
        """
        LayerTransmorph.run(self, datasets)
        X_final = self.merging.transform()
        self.output_data = []
        offset = 0
        for dataset in self.datasets:
            n_obs = dataset.shape[0]
            self.output_data.append(X_final[offset : offset + n_obs])
            offset += n_obs


class LayerChecking(LayerTransmorph):
    """
    TODO
    """

    def __init__(self, checking) -> None:
        super().__init__(
            type_="checking",
            compatible_types=["matching", "output"],
            maximum_inputs=1,
        )
        self.valid = False
        self.checking = checking

    def connect(self, layer: LayerTransmorph):
        """
        LayerChecking needs to specify output connection types.
        """
        raise NotImplementedError

    def connect_invalid(self, layer: LayerTransmorph):
        """
        TODO
        """
        self.check_connection(layer)
        # TODO

    def connect_valid(self, layer: LayerTransmorph):
        """
        TODO
        """
        self.check_connection(layer)
        # TODO

    def run(self, datasets: List[AnnData]):
        """
        TODO
        """
        if self.computed:
            return
        assert self.input_layers[0].output_data is not None
        input_data = self.input_layers[0].output_data.copy()
        self.valid = self.checking.check(input_data)
        while not self.valid:
            self.output_layers[0].set_computed(False)
            self.output_layers[0].input_layers[0].output_data = input_data
            pass
        self.output_data = self.input_layers[0].output_data
        self.set_computed(True)


class LayerOutput(LayerTransmorph):
    """
    Simple layer to manage network outputs. There can be several output layers.

    Possible inputs: 1
    ------------------
    - LayerMerging
    - LayerPreprocessing
    - LayerChecking

    Possible outputs: 0
    -------------------
    """

    output_id = 0

    def __init__(self) -> None:
        super().__init__("output", [], 0)
        self.id = LayerOutput.output_id
        LayerOutput.output_id += 1

    def run(self, datasets: List[AnnData]):
        """
        Runs the upstream pipeline and stores results in AnnData objects.
        """
        self.output_data = self.input_layers[0].output_data.copy()
        offset = 0
        X_int = self.output_data
        for adata in datasets:
            n_obs = adata.n_obs
            adata.obsm[f"X_transmorph_{self.id}"] = X_int[offset : offset + n_obs]


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

    def __init__(self) -> None:
        self.input_layer = None
        self.output_layers = []
        self.layers = []

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
        self.input_layer = input_layer
        layers_to_visit: List[LayerTransmorph] = [self.input_layer]
        while len(layers_to_visit) > 0:
            current_layer = layers_to_visit.pop(0)
            self.layers.append(current_layer)
            for output_layer in current_layer.iter_outputs():
                if output_layer in self.layers:
                    continue
                if type(output_layer) is LayerOutput:
                    self.output_layers.append(output_layer)
                layers_to_visit.append(output_layer)
                self.layers.append(output_layer)
        assert len(self.output_layers) > 0, "No output layer reachable from input."

    def run(self, datasets: List[AnnData]):
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
        self.input_layer.set_computed(False)
        for output_layer in self.output_layers:
            output_layer.run(datasets)
