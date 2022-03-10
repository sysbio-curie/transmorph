#!/usr/bin/env python3

from transmorph.checking.checkingTest import CheckingTest
from transmorph.datasets import load_spirals
from transmorph.matching import MatchingMNN
from transmorph.merging import MergingBarycenter
from transmorph import (
    LayerChecking,
    LayerInput,
    LayerMatching,
    LayerMerging,
    LayerOutput,
    TransmorphPipeline,
)
from anndata import AnnData

verbose = True

layer_input = LayerInput(verbose)
layer_match = LayerMatching(MatchingMNN(), verbose)
layer_merge = LayerMerging(MergingBarycenter(), verbose)
layer_check = LayerChecking(CheckingTest(), n_checks_max=3, verbose=verbose)
layer_out = LayerOutput(verbose)
layer_input.connect(layer_match)
layer_match.connect(layer_merge)
layer_merge.connect(layer_check)
layer_check.connect_no(layer_match)
layer_check.connect_yes(layer_out)

pipeline = TransmorphPipeline(verbose)
pipeline.initialize(layer_input)

xs, yt = load_spirals()
datasets = [AnnData(xs), AnnData(yt)]
pipeline.fit(datasets, reference=datasets[1])
print("# adata.obsm['transmorph] #")
print([adata.obsm["transmorph"].shape for adata in datasets])
print("# Remaining intermediate results #")
print([adata.uns["transmorph"] for adata in datasets])
