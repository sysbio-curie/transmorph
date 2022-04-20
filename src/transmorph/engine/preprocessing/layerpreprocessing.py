#!/usr/bin/env python3

import logging

from anndata import AnnData
from typing import List

from transmorph.engine import Layer
from transmorph.engine.preprocessing import IsPreprocessable
from transmorph.engine.profiler import IsProfilable, profile_method
from transmorph.engine.traits import IsRepresentable
from transmorph.engine.watchers import IsWatchable, WatcherTiming


class LayerPreprocessing(
    Layer, IsPreprocessable, IsWatchable, IsProfilable, IsRepresentable
):
    """
    This layer encapsulates a series of preprocessing algorithms derived
    from PreprocessingABC.
    """

    def __init__(self) -> None:
        Layer.__init__(
            self,
            compatible_inputs=[IsRepresentable],
            str_identifier="PREPROCESSING",
        )
        IsWatchable.__init__(self, compatible_watchers=[WatcherTiming])
        IsRepresentable.__init__(self, repr_key=f"{self}_{self.layer_id}")

    @profile_method
    def fit(self, datasets: List[AnnData]) -> List[Layer]:
        """
        Simply runs preprocessing algorithms and returns the result.
        """
        if self.has_preprocessings:
            self.log("Calling preprocessings.", level=logging.INFO)
        Xs = self.preprocess(datasets, self.embedding_reference)
        for adata, X_after in zip(datasets, Xs):
            self.write(adata, X_after)
        self.log("Done.", level=logging.INFO)
        return self.output_layers
