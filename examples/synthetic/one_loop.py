#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

from transmorph.datasets import load_spirals
from transmorph.engine import (
    LayerChecking,
    LayerInput,
    LayerMatching,
    LayerMerging,
    LayerOutput,
    TransmorphPipeline,
)

from transmorph.checking import CheckingEntropy
from transmorph.matching import MatchingMNN
from transmorph.merging import MergingLinearCorrection

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
lout = LayerOutput(verbose=VERBOSE)

linput.connect(lmatch)
lmatch.connect(lmerge)
lmerge.connect(lcheck)
lcheck.connect_yes(lout)
lcheck.connect_no(lmatch)

pipeline = TransmorphPipeline(verbose=VERBOSE)
pipeline.initialize(linput)

# Running the pipeline

spirals_data = load_spirals()
adata1, adata2 = spirals_data["src"], spirals_data["ref"]
pipeline.fit([adata1, adata2], reference=adata2)

# Retrieving and displaying results in a PC plot

pca = PCA(n_components=2)
X2 = pca.fit_transform(adata2.obsm["transmorph"])
X1 = pca.transform(adata1.X)
X1_int = pca.transform(adata1.obsm["transmorph"])

plt.figure(figsize=(6, 6))
plt.scatter(*X2.T, label="Reference dataset")
plt.scatter(*X1.T, label="Source dataset")
plt.scatter(*X1_int.T, label="Integrated dataset")
plt.legend()
plt.xticks([])
plt.yticks([])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("I >(1) MNN > LC > Chk >(1) O")
plt.savefig(f"{os.getcwd()}/transmorph/examples/synthetic/figures/one_loop.png")
plt.show()
