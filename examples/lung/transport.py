#!/usr/bin/env python3

from transmorph.datasets import load_travaglini_10x
from transmorph.preprocessing import PPStandardize, PPPCA
from transmorph.recipes import Transport
from transmorph.subsampling import SubsamplingVertexCover

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
import umap

# Building a subsampling pipeline
# Input -> PP -> MatchMNN + VertexCover -> MergeBarycenter -> Output

VERBOSE = True

subsampling = SubsamplingVertexCover(n_neighbors=10)
pipeline = Transport(
    flavor="emd",
    subsampling=subsampling,
    preprocessing=[PPStandardize(), PPPCA(n_components=30)],
    verbose=True,
)

# Running the pipeline

datasets = load_travaglini_10x()
adatas = (
    datasets["patient_1"],
    datasets["patient_2"],
    datasets["patient_3"],
)
pipeline.fit(adatas, reference=adatas[2])

# Plotting the result

all_X = np.concatenate([adata.obsm["transmorph"] for adata in adatas])
all_X_umap = umap.UMAP(min_dist=0.5).fit_transform(
    PCA(n_components=30).fit_transform(all_X)
)

representations = []
offset = 0
for adata in adatas:
    nobs = adata.n_obs
    representations.append(all_X_umap[offset : offset + nobs])
    offset += nobs

plt.figure()
ctypes = set(adatas[0].obs["cell_type"])
colors = ["orange", "royalblue", "darkgreen", "purple"]
label_names = ["Endothelial", "Stromal", "Epithelial", "Immune"]
plt_kwargs = {"ec": "k", "s": 40}

for ctype, c, l in zip(ctypes, colors, label_names):
    legend = True
    for X, adata in zip(representations, adatas):
        if legend:
            plt.scatter(
                *X[adata.obs["cell_type"] == ctype].T,
                label=l,
                c=c,
                **plt_kwargs,
            )
            legend = False
        else:
            plt.scatter(
                *X[adata.obs["cell_type"] == ctype].T,
                c=c,
                **plt_kwargs,
            )
plt.legend()
plt.xticks([])
plt.yticks([])
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("I > EMD > Bary > O")
plt.savefig(f"{os.getcwd()}/transmorph/examples/lung/figures/transport_pertype.png")
plt.show()

plt.figure()
plt_kwargs = {"ec": "k", "s": 40}

for i, (adata, X) in enumerate(zip(adatas, representations)):
    plt.scatter(
        *X.T,
        label=f"Patient {i}",
        **plt_kwargs,
    )
plt.legend()
plt.xticks([])
plt.yticks([])
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("I > EMD > Bary > O")
plt.savefig(f"{os.getcwd()}/transmorph/examples/lung/figures/transport_perpatient.png")
plt.show()
