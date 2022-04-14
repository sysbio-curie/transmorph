#!/usr/bin/env python3

# Testing all matchings

from transmorph.datasets import load_zhou_10x
from transmorph.engine import (
    LayerInput,
    LayerPreprocessing,
    LayerMatching,
    WatcherMatching,
    WatcherTiming,
    TransmorphPipeline,
)
from transmorph.matching import MatchingEMD
from transmorph.preprocessing import PPCommonGenes, PPStandardize, PPPCA
from transmorph.subsampling import SubsamplingVertexCover

from os.path import dirname

# Building a subsampling pipeline
# Input -> PP -> MatchMNN + VertexCover -> MergeBarycenter -> Output

for n_top_genes in [500, 800, 1000, 2000, 3000, 4000, 5000, 8000, 10000]:
    subsampling = SubsamplingVertexCover(n_neighbors=10)
    ppcom = PPCommonGenes(n_top_var=n_top_genes, verbose=True)

    linput = LayerInput()
    lppcom = LayerPreprocessing(preprocessing=ppcom)
    lppstd = LayerPreprocessing(preprocessing=PPStandardize(True, True))
    lpppca = LayerPreprocessing(preprocessing=PPPCA(n_components=30))
    lmatch = LayerMatching(matching=MatchingEMD(subsampling=subsampling))

    linput.connect(lppcom)
    lppcom.connect(lppstd)
    lppstd.connect(lpppca)
    lpppca.connect(lmatch)

    watch1 = WatcherMatching(lmatch, "cell_type")
    watch2 = WatcherTiming(lmatch)

    pipeline = TransmorphPipeline(verbose=False)
    pipeline.initialize(linput)

    # Running the pipeline

    datasets = list(load_zhou_10x().values())
    pipeline.fit(datasets)

    with open(f"{dirname(__file__)}/logs/emd_accuracy_per_gene.txt", "a") as fout:
        ngenes = ppcom.n_genes
        s0 = watch1.data["#samples0"]
        s1 = watch1.data["#samples1"]
        s2 = watch1.data["#samples2"]
        _0v1 = watch1.data["0,1"]
        _0v2 = watch1.data["0,2"]
        _1v2 = watch1.data["1,2"]
        ti = watch2.data["time"]
        fout.write(f"{ngenes},{s0},{s1},{s2},{_0v1},{_0v2},{_1v2},{ti}\n")
