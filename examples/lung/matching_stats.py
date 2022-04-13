#!/usr/bin/env python3

# Testing all matchings

from transmorph.datasets import load_travaglini_10x
from transmorph.engine import (
    LayerInput,
    LayerPreprocessing,
    LayerMatching,
    WatcherMatching,
    WatcherTiming,
    TransmorphPipeline,
)
from transmorph.matching import MatchingMNN
from transmorph.preprocessing import PPCommonGenes, PPStandardize, PPPCA
from transmorph.subsampling import SubsamplingVertexCover

# Building a subsampling pipeline
# Input -> PP -> MatchMNN + VertexCover -> MergeBarycenter -> Output

for ngenes in [100, 200, 300, 500, 800, 1000, 2000, 3000, 4000, 5000, 8000, 10000]:
    subsampling = SubsamplingVertexCover(n_neighbors=10)

    linput = LayerInput()
    lppcom = LayerPreprocessing(
        preprocessing=PPCommonGenes(n_top_var=ngenes, verbose=True)
    )
    lppstd = LayerPreprocessing(preprocessing=PPStandardize(True, True))
    lpppca = LayerPreprocessing(preprocessing=PPPCA(n_components=30))
    lmatch = LayerMatching(matching=MatchingMNN(subsampling=subsampling))

    linput.connect(lppcom)
    lppcom.connect(lppstd)
    lppstd.connect(lpppca)
    lpppca.connect(lmatch)

    watch1 = WatcherMatching(lmatch, "compartment")
    watch2 = WatcherTiming(lmatch)

    pipeline = TransmorphPipeline(verbose=False)
    pipeline.initialize(linput)

    # Running the pipeline

    datasets = list(load_travaglini_10x().values())
    pipeline.fit(datasets)

    print(watch1.data, watch2.data)
