#!/usr/bin/env python3

from transmorph.datasets import load_spirals
from transmorph.engine import (
    LayerInput,
    LayerMatching,
    LayerMerging,
    LayerOutput,
    Model,
)
from transmorph.matching import MatchingMNN
from transmorph.merging import MergingMDI

from transmorph.utils.plotting import plot_result

# Building a simple pipeline
# Input -> MatchMNN -> MergeBarycenter -> Output

VERBOSE = True

linput = LayerInput(verbose=VERBOSE)
lmatch = LayerMatching(matching=MatchingMNN(), verbose=VERBOSE)
lmerge = LayerMerging(merging=MergingMDI(), verbose=VERBOSE)
lout = LayerOutput(verbose=VERBOSE)

linput.connect(lmatch)
lmatch.connect(lmerge)
lmerge.connect(lout)

pipeline = Model(verbose=VERBOSE)
pipeline.initialize(linput)

# Running the pipeline

spirals_data = load_spirals()
adata1, adata2 = spirals_data["src"], spirals_data["ref"]
pipeline.fit([adata1, adata2])

# Retrieving and displaying results in a PC plot

plot_result(
    datasets=[adata1, adata2],
    reducer="pca",
    color_by="label",
    title="I > MNN > MDI > O",
    xlabel="MDI1",
    ylabel="MDI2",
    show=False,
    save=True,
    caller_path=f"{__file__}",
)
