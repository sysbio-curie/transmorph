#!/usr/bin/env python3

from transmorph.datasets import load_travaglini_10x
from transmorph.preprocessing import PPStandardize, PPPCA
from transmorph.recipes import Transport
from transmorph.subsampling import SubsamplingVertexCover
from transmorph.utils import plot_result

# Building a subsampling pipeline
# Input -> PP -> MatchMNN + VertexCover -> MergeBarycenter -> Output

VERBOSE = True

subsampling = SubsamplingVertexCover(n_neighbors=10)
pipeline = Transport(
    flavor="emd",
    subsampling=subsampling,
    preprocessing=[PPStandardize(), PPPCA(n_components=30)],
    verbose=VERBOSE,
)

# Running the pipeline

adatas = list(load_travaglini_10x().values())
pipeline.fit(adatas, reference=adatas[1])

# Plotting the result


plot_result(
    datasets=adatas,
    title="Recipe: Transport",
    show=False,
    save=True,
    use_cache=True,
    caller_path=f"{__file__}",
)

plot_result(
    datasets=adatas,
    color_by="compartment",
    title="Recipe: Transport",
    show=False,
    save=True,
    use_cache=True,
    caller_path=f"{__file__}",
)
