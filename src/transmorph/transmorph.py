#!/usr/bin/env python3


from abc import abstractmethod
from typing import List

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
        maximum_outputs: int = -1,
    ) -> None:
        self.type = type_
        self.output_layers = []
        self.maximum_outputs = maximum_outputs
        self.compatible_types = compatible_types
        self.output_data = None

    def connect(self, t) -> None:
        assert type(t) in self.compatible_types, f"Error: Uncompatible type: {type(t)}"
        assert (
            len(self.output_layers) < self.maximum_outputs or self.maximum_outputs == -1
        ), f"Error: too many output layers. Maximum: {self.maximum_outputs}"
        self.output_layers.append(t)

    def iter_outputs(self):
        for layer in self.output_layers:
            yield layer

    @abstractmethod
    def run(self, input_layers):
        pass


class LayerInput(TransmorphLayer):
    """
    TODO
    """

    def __init__(self) -> None:
        super().__init__("input", [LayerMatching], 1)

    def load_datasets(self, datasets: List[TData]):
        """
        TODO
        """
        self.output_data = [d for d in datasets]

    def run(self, input_layers):
        """
        TODO
        """
        assert len(input_layers) == 0, "Input layers cannot have predecessors."


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

    def run(self, input_layers):
        """
        TODO
        """
        assert len(input_layers) == 1, "Matching layer must have only one input."
        input_layer = input_layers[0]
        T = self.matching.fit(input_layer.output_data)
        self.output_data = T


class LayerCombineMatching(TransmorphLayer):
    """
    TODO
    """

    def __init__(self) -> None:
        super().__init__(
            "combine_matching",
            [LayerMerging, LayerCombineMatching],
            1,
        )

    def run(self, input_layers):
        """
        TODO
        """
        assert len(input_layers) > 0, "At least one input layer required."
        assert all(
            type(layer) is LayerMatching or type(layer) is LayerCombineMatching
            for layer in input_layers
        ), "At least one input layer is not of Matching type."
        Ts = [layer.output_data for layer in input_layers]
        print(Ts)


class LayerMerging(TransmorphLayer):
    """
    TODO
    """

    def __init__(self) -> None:
        super().__init__("merging", [LayerMatching, LayerChecking])


class LayerChecking(TransmorphLayer):
    """
    TODO
    """

    def __init__(self) -> None:
        super().__init__("checking", [LayerMatching, LayerOutput], 2)


class LayerOutput(TransmorphLayer):
    """
    TODO
    """

    def __init__(self) -> None:
        super().__init__("output", [], 0)


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
        pass
