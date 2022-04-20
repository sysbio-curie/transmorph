#!/usr/bin/env python3

from anndata import AnnData
from typing import List, Optional

from ..engine import (
    LayerInput,
    LayerMatching,
    LayerMerging,
    LayerOutput,
    LayerPreprocessing,
    Model,
)

from ..engine.layers import LayerTransmorph
from ..matching.matchingABC import MatchingABC
from ..merging.mergingABC import MergingABC
from ..preprocessing.preprocessingABC import PreprocessingABC


class MatchMerge:
    """
    Simple two-steps recipe, combining a matching and a merging:

    Optional preprocessing -> Matching -> Merging

    Parameters
    ----------
    matching: MatchingABC
        Matching algorithm to use.

    merging: MergingABC
        Merging algorithm to use.

    preprocessing: List[PreprocessingABC], optional
        Sequence of preprocessing steps to carry out before the
        pipeline.
    """

    def __init__(
        self,
        matching: MatchingABC,
        merging: MergingABC,
        preprocessing: Optional[List[PreprocessingABC]] = None,
        verbose: bool = False,
    ):
        self.matching = matching
        self.merging = merging
        self.preprocessing = [] if preprocessing is None else preprocessing
        self.verbose = verbose

        self.layers: List[LayerTransmorph] = [LayerInput()]
        self.layers += [LayerPreprocessing(ppobj) for ppobj in self.preprocessing]
        self.layers.append(LayerMatching(matching=self.matching))
        self.layers.append(LayerMerging(merging=self.merging))
        self.layers.append(LayerOutput())

        current = self.layers[0]
        for nextl in self.layers[1:]:
            current.connect(nextl)
            current = nextl

        self.pipeline = Model(verbose=self.verbose)
        self.pipeline.initialize(self.layers[0])

    def fit(self, datasets: List[AnnData], reference=None):
        self.pipeline.fit(datasets, reference=reference)
