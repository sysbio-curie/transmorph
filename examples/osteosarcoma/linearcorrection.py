#!/usr/bin/env python3

from transmorph.checking import CheckingEntropy
from transmorph.datasets import load_zhou_10x
from transmorph.engine import (
    LayerChecking,
    LayerInput,
    LayerMatching,
    LayerMerging,
    LayerOutput,
    LayerPreprocessing,
    TransmorphPipeline,
)
from transmorph.matching import MatchingMNN
from transmorph.merging import MergingLinearCorrection
from transmorph.preprocessing import PPCommonGenes, PPStandardize, PPPCA
from transmorph.subsampling import SubsamplingVertexCover

from transmorph.utils.plotting import plot_result


# Building a subsampling pipeline
# Input -> PP -> MatchMNN + VertexCover -> MergeBarycenter -> Output

VERBOSE = True

subsampling1 = SubsamplingVertexCover(n_neighbors=10)
subsampling2 = SubsamplingVertexCover(n_neighbors=10)

linput = LayerInput(verbose=VERBOSE)
lppcom = LayerPreprocessing(preprocessing=PPCommonGenes(verbose=True), verbose=VERBOSE)
lppstd = LayerPreprocessing(preprocessing=PPStandardize(True, True), verbose=VERBOSE)
lpppca = LayerPreprocessing(preprocessing=PPPCA(n_components=30), verbose=VERBOSE)
lmatchmnn1 = LayerMatching(
    matching=MatchingMNN(subsampling=subsampling1), verbose=VERBOSE
)
lmergelc1 = LayerMerging(
    merging=MergingLinearCorrection(learning_rate=0.3), verbose=VERBOSE
)
lpppca2 = LayerPreprocessing(preprocessing=PPPCA(n_components=30), verbose=VERBOSE)
lchecking = LayerChecking(CheckingEntropy(), verbose=VERBOSE, n_checks_max=5)
lmatchmnn = LayerMatching(
    matching=MatchingMNN(subsampling=subsampling2), verbose=VERBOSE
)
lmergelc2 = LayerMerging(merging=MergingLinearCorrection(), verbose=VERBOSE)
lout = LayerOutput(verbose=VERBOSE)

linput.connect(lppcom)
lppcom.connect(lppstd)
lppstd.connect(lpppca)
lpppca.connect(lmatchmnn1)
lmatchmnn1.connect(lmergelc1)
lmergelc1.connect(lpppca2)
lpppca2.connect(lchecking)
lchecking.connect_yes(lmatchmnn)
lchecking.connect_no(lmatchmnn1)
lmatchmnn.connect(lmergelc2)
lmergelc2.connect(lout)

lmergelc2.set_embedding_reference(lppstd)

pipeline = TransmorphPipeline(verbose=VERBOSE)
pipeline.initialize(linput)

# Running the pipeline

datasets = load_zhou_10x()
adatas = list(datasets.values())
pipeline.fit(adatas, reference=datasets["BC6"])

# Plotting results

plot_result(
    datasets=adatas,
    title="Complex pipeline",
    xlabel="MDI1",
    ylabel="MDI2",
    show=False,
    save=True,
    caller_path=f"{__file__}",
    use_cache=True,
)

plot_result(
    datasets=adatas,
    color_by="cell_type",
    title="Complex pipeline",
    xlabel="MDI1",
    ylabel="MDI2",
    show=False,
    save=True,
    caller_path=f"{__file__}",
    use_cache=True,
)

plot_result(
    datasets=adatas,
    color_by="malignant",
    title="Complex pipeline",
    xlabel="MDI1",
    ylabel="MDI2",
    show=False,
    save=True,
    caller_path=f"{__file__}",
    use_cache=True,
)
