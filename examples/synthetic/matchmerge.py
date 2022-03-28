#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

from transmorph.datasets import load_spirals
from transmorph.matching import MatchingMNN
from transmorph.merging import MergingBarycenter
from transmorph.recipes import MatchMerge

# Building a simple pipeline

spirals_data = load_spirals()
adata1, adata2 = spirals_data["src"], spirals_data["ref"]

recipe = MatchMerge(
    matching=MatchingMNN(),
    merging=MergingBarycenter(),
    verbose=True,
)
recipe.fit([adata1, adata2], reference=adata2)

# Retrieving and displaying results in a PC plot

pca = PCA(n_components=2)
X2 = pca.fit_transform(adata2.obsm["transmorph"])
X1 = pca.transform(adata1.X)
X1_int = pca.transform(adata1.obsm["transmorph"])

plt.figure(figsize=(6, 6))
plt.scatter(*X2.T, label="Reference dataset", c=adata2.obs["label"], ec="k")
plt.scatter(*X1.T, label="Source dataset", c="silver")
plt.scatter(
    *X1_int.T, label="Integrated dataset", c=adata1.obs["label"], ec="k", marker="s"
)
plt.legend()
plt.xticks([])
plt.yticks([])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("I > MNN > Bary > O")
plt.savefig(f"{os.getcwd()}/transmorph/examples/synthetic/figures/simple.png")
plt.show()
