#!/usr/bin/env python3

from typing import List

import numpy as np

from ..transformation import Transformation
from ...traits.usescommonfeatures import UsesCommonFeatures


class CommonFeatures(Transformation, UsesCommonFeatures):
    """
    Puts anndata objects in their larger common gene space. Only acts
    on AnnData.X matrix, so this must be done very early in a pipeline.
    If you want to use it internally in an object (e.g. a matching),
    instead make this object inherit UsesCommonFeatures.
    """

    def __init__(self):
        Transformation.__init__(
            self,
            str_identifier="COMMON_FEATURES",
            preserves_space=False,
        )
        UsesCommonFeatures.__init__(self, mode="total")

    def transform(self, datasets: List[np.ndarray]) -> List[np.ndarray]:
        """
        Uses the power of UsesCommonFeatures trait to solve the case.
        """
        results = []
        for i, X in enumerate(datasets):
            results.append(self.slice_features(X, i))
        return results
