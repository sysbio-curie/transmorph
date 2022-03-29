#!/usr/bin/env python3

from transmorph.engine.layers import (
    LayerInput,
    LayerOutput,
    LayerPreprocessing,
    LayerMatching,
    LayerMerging,
    LayerChecking,
    LayerType,
)

from transmorph.checking.checkingTest import CheckingTest
from transmorph.matching import MatchingMNN
from transmorph.merging import MergingBarycenter
from transmorph.preprocessing import PPStandardize

import numpy as np

# [i, j]: whether i -> j is allowed
comp_table = np.array(
    [
        # I,O,PP,MT,MG,CH
        [0, 1, 1, 1, 0, 1],  # I
        [0, 0, 0, 0, 0, 0],  # O
        [0, 1, 1, 1, 0, 1],  # PP
        [0, 0, 0, 0, 1, 0],  # MT
        [0, 1, 1, 1, 0, 1],  # MG
        [0, 1, 1, 1, 1, 1],  # C
    ]
)


def test_layers_compatibility_connection():
    example_layers = [
        LayerInput(),
        LayerOutput(),
        LayerPreprocessing(preprocessing=PPStandardize()),
        LayerMatching(matching=MatchingMNN()),
        LayerMerging(merging=MergingBarycenter()),
        LayerChecking(checking=CheckingTest()),
    ]
    for in_type in LayerType:
        for out_type in LayerType:
            if in_type < 0 or out_type < 0:
                continue  # Skip "BASE"
            out_layer = example_layers[out_type]
            if comp_table[in_type, out_type]:
                assert in_type in out_layer.get_compatible_inputs()
            else:
                assert in_type not in out_layer.get_compatible_inputs()


def test_comprehensive_pipeline():
    # We build a large pipeline with all layer types
    pass


if __name__ == "__main__":
    test_layers_compatibility_connection()
