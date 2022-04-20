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
from transmorph.merging import MergingBarycenter

from transmorph.utils import pca_multi, plot_result

# Building a simple pipeline
# Input -> MatchMNN -> MergeBarycenter -> Output

VERBOSE = True

linput = LayerInput(verbose=VERBOSE)
lmatch = LayerMatching(matching=MatchingMNN(), verbose=VERBOSE)
lmerge = LayerMerging(merging=MergingBarycenter(), verbose=VERBOSE)
lout = LayerOutput(verbose=VERBOSE)

linput.connect(lmatch)
lmatch.connect(lmerge)
lmerge.connect(lout)

pipeline = Model(verbose=VERBOSE)
pipeline.initialize(linput)

# Running the pipeline

spirals_data = load_spirals()
adata1, adata2 = spirals_data["src"], spirals_data["ref"]
P1, P2 = pca_multi(adata1.X, adata2.X)  # Simulating sc.pp.pca
adata1.obsm["X_pca"] = P1
adata2.obsm["X_pca"] = P2
pipeline.fit([adata1, adata2], reference=adata2, use_rep="X_pca")

# Retrieving and displaying results in a PC plot

plot_result(
    datasets=[adata1, adata2],
    reducer="pca",
    color_by="label",
    title="I (PC) >MNN > Bary > O",
    show=False,
    save=True,
    caller_path=f"{__file__}",
)
