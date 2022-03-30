#!/usr/bin/env python3

from anndata import AnnData
from typing import List


from ..engine import (
    LayerInput,
    LayerMatching,
    LayerMerging,
    LayerOutput,
    LayerPreprocessing,
    TransmorphPipeline,
)

from ..engine.layers import LayerTransmorph
from ..subsampling.subsamplingABC import SubsamplingABC

from ..matching import MatchingEMD, MatchingGW, MatchingSinkhorn
from ..merging import MergingBarycenter
from ..preprocessing.preprocessingABC import PreprocessingABC
from ..subsampling import SubsamplingKeepAll


class Transport:
    """
    Mimics SCOT, with a transport-based matching followed
    with a barycentric merging.

    https://pubmed.ncbi.nlm.nih.gov/35050714/
    """

    def __init__(
        self,
        flavor: str = "emd",
        subsampling: SubsamplingABC = SubsamplingKeepAll(),
        preprocessing: List[PreprocessingABC] = [],
        verbose: bool = False,
    ):
        self.flavor = flavor
        self.preprocessing = preprocessing
        self.verbose = verbose

        self.layers: List[LayerTransmorph] = [LayerInput(verbose=self.verbose)]
        self.layers += [
            LayerPreprocessing(ppobj, verbose=self.verbose)
            for ppobj in self.preprocessing
        ]
        if flavor == "emd":
            matching = MatchingEMD(subsampling=subsampling)
        elif flavor == "gromov":
            matching = MatchingGW(subsampling=subsampling)
        elif flavor == "sinkhorn":
            matching = MatchingSinkhorn(subsampling=subsampling)
        else:
            raise ValueError(
                f"Unrecognized flavor: {flavor}. Accepted parameters "
                "are 'emd', 'gromov' or 'sinkhorn'."
            )
        self.layers.append(LayerMatching(matching=matching, verbose=self.verbose))
        self.layers.append(
            LayerMerging(merging=MergingBarycenter(), verbose=self.verbose)
        )
        self.layers.append(LayerOutput(verbose=self.verbose))

        current = self.layers[0]
        for nextl in self.layers[1:]:
            current.connect(nextl)
            current = nextl

        self.pipeline = TransmorphPipeline(verbose=self.verbose)
        self.pipeline.initialize(self.layers[0])

    def fit(self, datasets: List[AnnData], reference):
        assert reference is not None, "Transport recipe requires a reference AnnData."
        self.pipeline.fit(datasets, reference=reference)
