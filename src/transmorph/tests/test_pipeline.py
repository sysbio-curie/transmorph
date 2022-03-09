#!/usr/bin/env python3

from transmorph.datasets import load_spirals
from transmorph.matching import MatchingMNN
from transmorph.merging import MergingBarycenter
from transmorph import (
    LayerInput,
    LayerMatching,
    LayerMerging,
    LayerOutput,
    TransmorphPipeline,
)
from anndata import AnnData

verbose = True

layer_input = LayerInput(verbose)
layer_match = LayerMatching(MatchingMNN(), verbose)
layer_merge = LayerMerging(MergingBarycenter(), verbose)
layer_out = LayerOutput(verbose)
layer_input.connect(layer_match)
layer_match.connect(layer_merge)
layer_merge.connect(layer_out)

pipeline = TransmorphPipeline(verbose)
pipeline.initialize(layer_input)

xs, yt = load_spirals()
datasets = [AnnData(xs), AnnData(yt)]
pipeline.fit(datasets, reference=datasets[1])
