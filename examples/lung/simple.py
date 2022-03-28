#!/usr/bin/env python3

from transmorph.datasets import load_travaglini_10x
from transmorph.engine import (
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
import os

# Building a subsampling pipeline
# Input -> PP -> MatchMNN + VertexCover -> MergeBarycenter -> Output

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

plt.figure()
ctypes = set(adata1.obs["cell_type"])
colors = ["orange", "royalblue", "darkgreen", "purple"]
label_names = ["Endothelial", "Stromal", "Epithelial", "Immune"]
plt_kwargs = {"ec": "k", "s": 40}

for ctype, c, l in zip(ctypes, colors, label_names):
    legend = True
    for adata in [adata1, adata2, adata3]:
        if legend:
            plt.scatter(
                *adata.obsm["transmorph"][adata.obs["cell_type"] == ctype].T,
                label=l,
                c=c,
                **plt_kwargs,
            )
            legend = False
        else:
            plt.scatter(
                *adata.obsm["transmorph"][adata.obs["cell_type"] == ctype].T,
                c=c,
                **plt_kwargs,
            )
plt.legend()
plt.xticks([])
plt.yticks([])
plt.xlabel("MDI1")
plt.ylabel("MDI2")
plt.savefig(f"{os.getcwd()}/transmorph/examples/lung/figures/simple_pertype.png")
plt.show()

plt.figure()
plt_kwargs = {"ec": "k", "s": 40}

for i, adata in enumerate([adata1, adata2, adata3]):
    plt.scatter(
        *adata.obsm["transmorph"].T,
        label=f"Patient {i}",
        **plt_kwargs,
    )
plt.legend()
plt.xticks([])
plt.yticks([])
plt.xlabel("MDI1")
plt.ylabel("MDI2")
plt.savefig(f"{os.getcwd()}/transmorph/examples/lung/figures/simple_perpatient.png")
plt.show()
