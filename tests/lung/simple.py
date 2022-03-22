#!/usr/bin/env python3
# TODO: this should work at this scale

from transmorph.datasets import load_travaglini_10x
from transmorph.layers import (
    LayerInput,
    LayerMatching,
    LayerMerging,
    LayerOutput,
    LayerPreprocessing,
    TransmorphPipeline,
)
from transmorph.matching import MatchingMNN
from transmorph.merging import MergingBarycenter
from transmorph.preprocessing import PPStandardize, PPPCA

# Building a simple pipeline
# Input -> MatchMNN -> MergeBarycenter -> Output

VERBOSE = True

linput = LayerInput(verbose=VERBOSE)
lppstd = LayerPreprocessing(preprocessing=PPStandardize(True, True), verbose=VERBOSE)
lpppca = LayerPreprocessing(preprocessing=PPPCA(n_components=30), verbose=VERBOSE)
lmatch = LayerMatching(matching=MatchingMNN(), verbose=VERBOSE)
lmerge = LayerMerging(merging=MergingBarycenter(), verbose=VERBOSE)
lout = LayerOutput(verbose=VERBOSE)

linput.connect(lppstd)
lppstd.connect(lpppca)
lpppca.connect(lmatch)
lmatch.connect(lmerge)
lmerge.connect(lout)

pipeline = TransmorphPipeline(verbose=VERBOSE)
pipeline.initialize(linput)

# Running the pipeline

datasets = load_travaglini_10x()
adata1, adata2, adata3 = (
    datasets["patient_1"],
    datasets["patient_2"],
    datasets["patient_3"],
)
pipeline.fit([adata1, adata2, adata3], reference=adata2)
