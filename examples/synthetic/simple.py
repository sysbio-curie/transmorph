#!/usr/bin/env python3

from transmorph.datasets import load_spirals
from transmorph.engine import (
    LayerInput,
    LayerMatching,
    LayerMerging,
    LayerOutput,
    TransmorphPipeline,
)
from transmorph.matching import MatchingMNN
from transmorph.merging import MergingBarycenter

from transmorph.utils.plotting import plot_result

# Building a simple pipeline
# Input -> MatchMNN -> MergeBarycenter -> Output

linput = LayerInput()
lmatch = LayerMatching(matching=MatchingMNN())
lmerge = LayerMerging(merging=MergingBarycenter())
lout = LayerOutput()

linput.connect(lmatch)
lmatch.connect(lmerge)
lmerge.connect(lout)

pipeline = TransmorphPipeline(verbose=True)
pipeline.initialize(linput)

# Running the pipeline

spirals_data = load_spirals()
adata1, adata2 = spirals_data["src"], spirals_data["ref"]
pipeline.fit([adata1, adata2], reference=adata2)

# Retrieving and displaying results in a PC plot

plot_result(
    datasets=[adata1, adata2],
    reducer="pca",
    color_by="label",
    title="I > MNN > Bary > O",
    show=False,
    save=True,
    caller_path=f"{__file__}",
)
