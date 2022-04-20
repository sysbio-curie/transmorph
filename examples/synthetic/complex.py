#!/usr/bin/env python3

from transmorph.datasets import load_spirals
from transmorph.engine import (
    LayerChecking,
    LayerInput,
    LayerMatching,
    LayerMerging,
    LayerOutput,
    Model,
)

from transmorph.checking import CheckingEntropy
from transmorph.matching import MatchingMNN, MatchingEMD
from transmorph.merging import MergingLinearCorrection, MergingBarycenter

from transmorph.utils.plotting import plot_result

# Building a simple pipeline
# Input -> MatchMNN -> MergeBarycenter -> Output

VERBOSE = True

linput = LayerInput(verbose=VERBOSE)
lmatch = LayerMatching(matching=MatchingMNN(), verbose=VERBOSE)
lmerge = LayerMerging(
    merging=MergingLinearCorrection(learning_rate=0.3), verbose=VERBOSE
)
lcheck = LayerChecking(
    checking=CheckingEntropy(threshold=0.5), n_checks_max=50, verbose=VERBOSE
)
lmatch_final = LayerMatching(matching=MatchingEMD(), verbose=VERBOSE)
lmerge_final = LayerMerging(merging=MergingBarycenter(), verbose=VERBOSE)
lout = LayerOutput(verbose=VERBOSE)

linput.connect(lmatch)
lmatch.connect(lmerge)
lmerge.connect(lcheck)
lcheck.connect_no(lmatch)
lcheck.connect_yes(lmatch_final)
lmatch_final.connect(lmerge_final)
lmerge_final.connect(lout)

pipeline = Model(verbose=VERBOSE)
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
    title="I >(1) MNN > LC > Chk >(1) EMD > MDI > O",
    show=False,
    save=True,
    caller_path=f"{__file__}",
)
