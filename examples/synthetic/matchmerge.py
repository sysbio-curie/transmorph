#!/usr/bin/env python3

from transmorph.datasets import load_spirals
from transmorph.matching import MatchingMNN
from transmorph.merging import MergingBarycenter
from transmorph.recipes import MatchMerge

from transmorph.utils.plotting import plot_result

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

plot_result(
    datasets=[adata1, adata2],
    reducer="pca",
    color_by="label",
    title="Recipe: MatchMerge (MNN/Bary)",
    show=False,
    save=True,
    caller_path=f"{__file__}",
)
