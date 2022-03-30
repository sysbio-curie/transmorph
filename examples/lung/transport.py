#!/usr/bin/env python3

from transmorph.datasets import load_travaglini_10x
from transmorph.preprocessing import PPStandardize, PPPCA
from transmorph.recipes import Transport
from transmorph.subsampling import SubsamplingVertexCover

from transmorph.utils.plotting import plot_result

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


plot_result(
    datasets=adatas,
    title="Recipe: Transport",
    show=False,
    save=True,
    caller_path=f"{__file__}",
)

plot_result(
    datasets=adatas,
    color_by="cell_type",
    title="Recipe: Transport",
    show=False,
    save=True,
    caller_path=f"{__file__}",
)
