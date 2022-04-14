#!/usr/bin/env python3

from transmorph.datasets import load_chen_10x
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

subsampling = SubsamplingVertexCover(n_neighbors=10)

linput = LayerInput()
lppstd = LayerPreprocessing(preprocessing=PPStandardize(True, True))
lmatch = LayerMatching(matching=MatchingMNN(subsampling=subsampling))
lmerge = LayerMerging(merging=MergingMDI())
lout = LayerOutput()

linput.connect(lppstd)
lppstd.connect(lmatch)
lmatch.connect(lmerge)
lmerge.connect(lout)

lmatch.add_preprocessing(PPCommonGenes(n_top_var=3000))
lmatch.add_preprocessing(PPPCA(n_components=30))
lmerge.add_preprocessing(PPPCA(n_components=30, strategy="independent"))

pipeline = TransmorphPipeline(verbose=True)
pipeline.initialize(linput)

# Running the pipeline

datasets = list(load_chen_10x().values())
pipeline.fit(datasets)

# Plotting results

plot_result(
    datasets=datasets,
    title="I > ST > PC > MNN (+VC) > MDI > O",
    xlabel="MDI1",
    ylabel="MDI2",
    show=False,
    save=True,
    caller_path=f"{__file__}",
)

plot_result(
    datasets=datasets,
    color_by="cell_type",
    title="I > ST > PC > MNN (+VC) > MDI > O",
    xlabel="MDI1",
    ylabel="MDI2",
    show=False,
    save=True,
    caller_path=f"{__file__}",
)