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
from transmorph.merging import MergingMDI
from transmorph.preprocessing import PPStandardize, PPPCA
from transmorph.subsampling import SubsamplingVertexCover

import matplotlib.pyplot as plt

# Building a simple pipeline
# Input -> PP -> -MatchMNN -> MergeBarycenter -> Output

VERBOSE = True

subsampling = SubsamplingVertexCover(n_neighbors=10)

linput = LayerInput(verbose=VERBOSE)
lppstd = LayerPreprocessing(preprocessing=PPStandardize(True, True), verbose=VERBOSE)
lpppca = LayerPreprocessing(preprocessing=PPPCA(n_components=30), verbose=VERBOSE)
lmatch = LayerMatching(matching=MatchingMNN(subsampling=subsampling), verbose=VERBOSE)
lmerge = LayerMerging(merging=MergingMDI(), verbose=VERBOSE)
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

ctypes = set(adata1.obs["cell_type"])

for ctype in ctypes:
    for adata in [adata1, adata2, adata3]:
        plt.scatter(*adata.obsm["transmorph"][adata.obs["cell_type"] == ctype].T)
plt.show()
