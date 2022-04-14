#!/usr/bin/env python3

# Testing all matchings

from transmorph.datasets import load_chen_10x
from transmorph.engine import (
    LayerInput,
    LayerPreprocessing,
    LayerMatching,
    WatcherMatching,
    WatcherTiming,
    TransmorphPipeline,
)
from transmorph.matching import MatchingEMD, MatchingMNN
from transmorph.preprocessing import PPCommonGenes, PPStandardize, PPPCA
from transmorph.subsampling import SubsamplingVertexCover

from os.path import dirname

subsampling = SubsamplingVertexCover(n_neighbors=10)
method_names = ["mnn_exact", "mnn_louvain", "emd"]
methods = [
    MatchingMNN(subsampling=subsampling, algorithm="exact"),
    MatchingMNN(subsampling=subsampling, algorithm="louvain"),
    MatchingEMD(subsampling=subsampling),
]

datasets = list(load_chen_10x().values())
LABEL = "cell_type"

for methname, method in zip(method_names, methods):
    print(f"## Method: {methname}")
    FNAME = f"{dirname(__file__)}/logs/{methname}_accuracy_per_gene.txt"

    with open(FNAME, "w") as fout:
        fout.write(
            "#NGENES,N0,N1,N2,ACC_0V1,ACC_0V2,ACC_1V2,"
            "ACCi_0v1,ACCi_0v2,ACCi_1v2,TIME\n"
        )

    for n_top_genes in [
        500,
        1000,
        2000,
        3000,
        4000,
        5000,
        6000,
        7000,
        8000,
        9000,
        10000,
    ]:
        print(f"#### Top genes: {n_top_genes}")
        ppcom = PPCommonGenes(n_top_var=n_top_genes, verbose=True)

        linput = LayerInput()
        lppcom = LayerPreprocessing(preprocessing=ppcom)
        lppstd = LayerPreprocessing(preprocessing=PPStandardize(True, True))
        lpppca = LayerPreprocessing(preprocessing=PPPCA(n_components=30))
        lmatch = LayerMatching(matching=method)

        linput.connect(lppcom)
        lppcom.connect(lppstd)
        lppstd.connect(lpppca)
        lpppca.connect(lmatch)

        watch1 = WatcherMatching(target=lmatch, label=LABEL, ignore_unmatched=False)
        watch2 = WatcherMatching(target=lmatch, label=LABEL, ignore_unmatched=True)
        watch3 = WatcherTiming(lmatch)

        pipeline = TransmorphPipeline(verbose=False)
        pipeline.initialize(linput)

        # Running the pipeline
        try:
            pipeline.fit(datasets)
        except AssertionError:
            print(f"n_genes = {n_top_genes}, no common gene. Continuing.")
            continue

        with open(FNAME, "a") as fout:
            ngenes = ppcom.n_genes
            s0 = watch1.data["#samples0"]
            s1 = watch1.data["#samples1"]
            s2 = watch1.data["#samples2"]
            _0v1 = watch1.data["0,1"]
            _0v2 = watch1.data["0,2"]
            _1v2 = watch1.data["1,2"]
            _0v1i = watch2.data["0,1"]
            _0v2i = watch2.data["0,2"]
            _1v2i = watch2.data["1,2"]
            ti = watch2.data["time"]
            fout.write(
                f"{ngenes},{s0},{s1},{s2},{_0v1},{_0v2},"
                f"{_1v2},{_0v1i},{_0v2i},{_1v2i},{ti}\n"
            )
