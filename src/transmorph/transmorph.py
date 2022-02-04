#!/usr/bin/env python3


from abc import abstractmethod
from typing import List

from scipy.sparse.csr import csr_matrix

from transmorph.TData import TData
from transmorph.matching.matchingABC import MatchingABC


class TransmorphLayer:
    """
    TODO
    """

    def __init__(
        self,
        type_: str = "undefined",
        compatible_types=[],
        maximum_inputs: int = -1,
        maximum_outputs: int = -1,
    ) -> None:
        self.type = type_
        self.input_layers = []
        self.output_layers = []
        self.maximum_inputs = maximum_inputs
        self.maximum_outputs = maximum_outputs
        self.compatible_types = compatible_types
        self.output_data = None

    def connect(self, t) -> None:
        assert type(t) in self.compatible_types, f"Error: Uncompatible type: {type(t)}"
        assert (
            len(self.output_layers) < self.maximum_outputs or self.maximum_outputs == -1
        ), f"Error: too many output layers. Maximum: {self.maximum_outputs}"
        assert (
            len(t.input_layers) < t.maximum_outputs or t.maximum_outputs == -1
        ), f"Error: too many input layers. Maximum: {t.maximum_outputs}"
        self.output_layers.append(t)
        t.input_layers.append(self)

    def iter_outputs(self):
        for layer in self.output_layers:
            yield layer

    @abstractmethod
    def run(self, datasets):
        for layer in self.input_layers:
            layer.run()


class LayerInput(TransmorphLayer):
    """
    TODO
    """

    def __init__(self) -> None:
        super().__init__("input", [LayerMatching], 0, -1)

    def run(self, datasets):
        """
        TODO
        """
        self.output_data = [d for d in datasets]


class LayerMatching(TransmorphLayer):
    """
    TODO
    """

    def __init__(self, matching: MatchingABC) -> None:
        super().__init__(
            "matching",
            [LayerMerging, LayerCombineMatching],
            1,
        )
        self.matching = matching

    def get_datasets(self):
        return self.matching.datasets

    def run(self, datasets):
        """
        TODO
        """
        TransmorphLayer.run(self, datasets)
        input_layer = self.input_layers[0]
        T = self.matching.fit(input_layer.output_data)
        self.output_data = T


class LayerCombineMatching(TransmorphLayer):
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

    def get_datasets(self):
        return self.datasets

    def normalize(self, T_matching):
        """
        TODO
        """
        T_matching = csr_matrix(T_matching / T_matching.sum(axis=1))
        return T_matching + T_matching.T - T_matching.multiply(T_matching)

    def run(self, datasets):
        """
        TODO
        """
        TransmorphLayer.run(self, datasets)
        input_layers = self.input_layers
        assert all(
            type(layer) is LayerMatching or type(layer) is LayerCombineMatching
            for layer in input_layers
        ), "At least one input layer is not of Matching type."
        Ts = [self.normalize(layer.output_data.copy()) for layer in input_layers]
        self.output_data = Ts[0]
        if self.mode == "additive":
            for T in Ts[1:]:
                self.output_data += T
        if self.mode == "multiplicative":
            for T in Ts[1:]:
                self.output_data = self.output_data.multiply(T)
        if self.mode == "intersection":
            for T in Ts[1:]:
                self.output_data = self.output_data.minimum(T)
        self.output_data = self.normalize(self.output_data)
        self.datasets = input_layers[0].get_datasets()


class LayerMerging(TransmorphLayer):
    """
    TODO
    """

    def __init__(self, merging, datasets) -> None:
        super().__init__("merging", [LayerMatching, LayerChecking, LayerOutput])
        self.datasets = datasets
        self.merging = merging

    def run(self, datasets):
        TransmorphLayer.run(self, datasets)
        X_final = self.merging.transform()  # TODO: link with previous matching
        self.output_data = []
        offset = 0
        for dataset in self.datasets:
            n_obs = dataset.n_obs
            self.output_data.append(X_final[offset : offset + n_obs])
            offset += n_obs


class LayerChecking(TransmorphLayer):
    """
    TODO
    """

    def __init__(self, checking) -> None:
        super().__init__("checking", [LayerMatching, LayerOutput], 2)
        self.valid = False
        self.checking = checking

    def run(self, datasets):
        self.input_layers[0].run(datasets)
        self.valid = self.checking.check(self.input_layers[0].output_data.copy())
        if not self.valid:
            previous_input = self.output_layers[
                0
            ].input_layers  # We temporarily rewire the loop
            self.output_layers[0].input_layers = [self]
            while not self.valid:
                self.output_data = self.input_layers[0].output_data
                self.input_layers[0].run(datasets)
            self.output_layers[0].input_layers = previous_input  # We restore the links
        self.output_data = self.input_layers[0].output_data


class LayerOutput(TransmorphLayer):
    """
    TODO
    """

    def __init__(self) -> None:
        super().__init__("output", [], 0)

    def run(self, datasets):
        TransmorphLayer.run(self, datasets)
        self.output_data = self.input_layers[0].output_data.copy()


class TransmorphPipeline:
    """
    TODO
    """

    def __init__(self) -> None:
        self.input_layer = None
        self.output_layers = []
        self.layers = []

    def initialize(self, layers: List[TransmorphLayer]):
        """
        TODO
        """
        self.input_layer = None
        for layer in layers:
            if type(layer) is LayerInput:
                assert self.input_layer is not None, "Several input layers found."
                self.input_layer = layer
        assert self.input_layer is not None, "No input layer found."
        layers_to_visit = [self.input_layer]
        while len(layers_to_visit) > 0:
            current_layer = layers_to_visit.pop(0)
            for output_layer in current_layer.iter_outputs():
                if output_layer in self.layers or output_layer is layers_to_visit:
                    continue
                if type(output_layer) is LayerOutput:
                    self.output_layers.append(output_layer)
                layers_to_visit.append(output_layer)
        assert len(self.output_layers) > 0, "No output layer reachable from input."

    def run(self, datasets: List[TData]):
        """
        TODO
        """
        for output_layer in self.output_layers:
            output_layer.run()
