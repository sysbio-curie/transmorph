#!/usr/bin/env python3


from typing import List

from transmorph.TData import TData


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

    def connect(self, t) -> None:
        assert type(t) in self.compatible_types, f"Error: Uncompatible type: {type(t)}"
        assert (
            len(self.output_layers) < self.maximum_outputs or self.maximum_outputs == -1
        ), f"Error: too many output layers. Maximum: {self.maximum_outputs}"
        self.output_layers.append(t)


class LayerMatching(TransmorphLayer):
    """
    TODO
    """

    def __init__(self) -> None:
        super().__init__(
            "matching",
            [LayerMerging, LayerCombineMatching],
            1,
        )


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


class LayerInput(TransmorphLayer):
    """
    TODO
    """

    def __init__(self) -> None:
        super().__init__("input", [LayerMatching], 1)


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
        pass

    def initialize(self, layers: List[TransmorphLayer]):
        pass

    def run(self, datasets: List[TData]):
        pass
