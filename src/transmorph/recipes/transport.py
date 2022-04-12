#!/usr/bin/env python3

from anndata import AnnData
from typing import List, Optional
from warnings import warn

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
from ..merging import MergingBarycenter, MergingLinearCorrection
from ..preprocessing.preprocessingABC import PreprocessingABC


class Transport:
    """
    Mimics SCOT, with a transport-based matching followed
    with a barycentric merging.

    https://pubmed.ncbi.nlm.nih.gov/35050714/
    """

    def __init__(
        self,
        transport_flavor: str = "emd",
        merge_flavor: str = "linearcorrection",
        subsampling: Optional[SubsamplingABC] = None,
        preprocessing: List[PreprocessingABC] = [],
        verbose: bool = False,
    ):
        self.transport_flavor = transport_flavor
        self.merge_flavor = merge_flavor
        self.preprocessing = preprocessing
        self.verbose = verbose

        if transport_flavor == "emd":
            matching = MatchingEMD(subsampling=subsampling)
        elif transport_flavor == "gromov":
            matching = MatchingGW(subsampling=subsampling)
        elif transport_flavor == "sinkhorn":
            matching = MatchingSinkhorn(subsampling=subsampling)
        else:
            raise ValueError(
                f"Unrecognized flavor: {transport_flavor}. Accepted parameters "
                "are 'emd', 'gromov' or 'sinkhorn'."
            )
        if merge_flavor == "linearcorrection":
            merging = MergingLinearCorrection()
        elif merge_flavor == "barycenter":
            merging = MergingBarycenter()
            if subsampling is not None:
                warn(
                    "Subsampling specified, you should use merging"
                    "MergingLinearCorrection to avoid instabilities."
                )
        else:
            raise ValueError(
                f"Unrecognized flavor: {merge_flavor}. Accepted parameters "
                "are 'barycenter' or 'linearcorrection'."
            )

        self.layers: List[LayerTransmorph] = [LayerInput()]
        self.layers += [LayerPreprocessing(ppobj) for ppobj in self.preprocessing]
        self.layers.append(LayerMatching(matching=matching))
        self.layers.append(LayerMerging(merging=merging))
        self.layers.append(LayerOutput())

        current = self.layers[0]
        for nextl in self.layers[1:]:
            current.connect(nextl)
            current = nextl

        self.pipeline = TransmorphPipeline(verbose=self.verbose)
        self.pipeline.initialize(self.layers[0])

    def fit(self, datasets: List[AnnData], reference):
        assert reference is not None, "Transport recipe requires a reference AnnData."
        self.pipeline.fit(datasets, reference=reference)
