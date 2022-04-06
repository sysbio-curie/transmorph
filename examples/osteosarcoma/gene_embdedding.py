#!/usr/bin/env python3

from transmorph.datasets import load_zhou_10x
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
from transmorph.preprocessing import PPCommonGenes, PPStandardize, PPPCA
from transmorph.subsampling import SubsamplingVertexCover

from transmorph.utils.plotting import plot_result


# Building a subsampling pipeline
# Input -> PP -> MatchMNN + VertexCover -> MergeBarycenter -> Output

VERBOSE = True

subsampling = SubsamplingVertexCover(n_neighbors=10)

linput = LayerInput(verbose=VERBOSE)
lppcom = LayerPreprocessing(
    preprocessing=PPCommonGenes(verbose=VERBOSE), verbose=VERBOSE
)
lppstd = LayerPreprocessing(preprocessing=PPStandardize(True, True), verbose=VERBOSE)
lpppca = LayerPreprocessing(preprocessing=PPPCA(n_components=30), verbose=VERBOSE)
lmatch = LayerMatching(matching=MatchingMNN(subsampling=subsampling), verbose=VERBOSE)
lmerge = LayerMerging(merging=MergingMDI(), verbose=VERBOSE)
lout = LayerOutput(verbose=VERBOSE)

linput.connect(lppcom)
lppcom.connect(lppstd)
lppstd.connect(lpppca)
lpppca.connect(lmatch)
lmatch.connect(lmerge)
lmerge.connect(lout)

# merge in gene space
lmerge.set_embedding_reference(lppstd)

pipeline = TransmorphPipeline(verbose=VERBOSE)
pipeline.initialize(linput)

# Running the pipeline

datasets = load_zhou_10x()
adatas = list(datasets.values())
pipeline.fit(adatas)

# Plotting results

plot_result(
    datasets=adatas,
    title="I > ST > PC > MNN (+VC) > LC > O",
    xlabel="UMAP1",
    ylabel="UMAP2",
    show=False,
    save=True,
    caller_path=f"{__file__}",
    use_cache=True,
)

plot_result(
    datasets=adatas,
    color_by="cell_type",
    title="I > ST > PC > MNN (+VC) > LC > O",
    xlabel="UMAP1",
    ylabel="UMAP2",
    show=False,
    save=True,
    caller_path=f"{__file__}",
    use_cache=True,
)
