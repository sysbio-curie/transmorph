#!/usr/bin/env python3

from transmorph.datasets import load_travaglini_10x
from transmorph.recipes import MatchMerge

from transmorph.matching import MatchingPartialOT
from transmorph.merging import MergingBarycenter
from transmorph.preprocessing import PPStandardize, PPPCA
from transmorph.subsampling import SubsamplingVertexCover

from transmorph.utils import plot_result

pipeline = MatchMerge(
    matching=MatchingPartialOT(
        transport_mass=0.7, subsampling=SubsamplingVertexCover()
    ),
    merging=MergingBarycenter(),
    preprocessing=[PPStandardize(center=True, scale=True), PPPCA(n_components=30)],
    verbose=True,
)

datasets = list(load_travaglini_10x().values())
pipeline.fit(datasets, reference=datasets[1])

plot_result(
    datasets=datasets,
    title="Recipe: MatchMerge",
    show=False,
    save=True,
    use_cache=True,
    caller_path=f"{__file__}",
)

plot_result(
    datasets=datasets,
    title="Recipe: MatchMerge",
    color_by="compartment",
    show=False,
    save=True,
    use_cache=True,
    caller_path=f"{__file__}",
)
