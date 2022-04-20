#!/usr/bin/env python3

from typing import List

import numpy as np

from transmorph.engine.preprocessing import Preprocessing
from transmorph.engine.traits import UsesCommonFeatures


class CommonGenes(Preprocessing, UsesCommonFeatures):
    """
    Puts anndata objects in their larger common gene space. Only acts
    on AnnData.X matrix, so this must be done very early in a pipeline.
    """

    def __init__(self):
        Preprocessing.__init__(self, str_identifier="COMMON_GENES")
        UsesCommonFeatures.__init__(self, mode="total")

    def transform(self, datasets: List[np.ndarray]) -> List[np.ndarray]:
        """
        Uses the power of UsesCommonFeatures trait to solve the case.
        """
        results = []
        for i, X in enumerate(datasets):
            results.append(self.slice_features(X, i))
        return results
