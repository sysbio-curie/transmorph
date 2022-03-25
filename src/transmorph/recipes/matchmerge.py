#!/usr/bin/env python3

from anndata import AnnData
from typing import List

from transmorph.layers.layers import LayerTransmorph

from ..layers import (
    LayerInput,
    LayerMatching,
    LayerMerging,
    LayerOutput,
    LayerPreprocessing,
    TransmorphPipeline,
)
from ..matching.matchingABC import MatchingABC
from ..merging.mergingABC import MergingABC
from ..preprocessing.preprocessingABC import PreprocessingABC


class MatchMerge:
    """ """

    def __init__(
        self,
        matching: MatchingABC,
        merging: MergingABC,
        preprocessing: List[PreprocessingABC] = [],
        verbose: bool = False,
    ):
        self.matching = matching
        self.merging = merging
        self.preprocessing = preprocessing
        self.verbose = verbose

        self.layers: List[LayerTransmorph] = [LayerInput(verbose=self.verbose)]
        self.layers += [
            LayerPreprocessing(ppobj, verbose=self.verbose)
            for ppobj in self.preprocessing
        ]
        self.layers.append(LayerMatching(matching=self.matching, verbose=self.verbose))
        self.layers.append(LayerMerging(merging=self.merging, verbose=self.verbose))
        self.layers.append(LayerOutput(verbose=self.verbose))

        current = self.layers[0]
        for nextl in self.layers[1:]:
            current.connect(nextl)
            current = nextl

        self.pipeline = TransmorphPipeline(verbose=self.verbose)
        self.pipeline.initialize(self.layers[0])

    def fit(self, datasets: List[AnnData], reference=None):
        self.pipeline.fit(datasets, reference=reference)
