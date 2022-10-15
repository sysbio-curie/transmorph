#!/usr/bin/env python3

from typing import List

import anndata as ad
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

    def transform(
        self,
        datasets: List[ad.AnnData],
        embeddings: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Uses the power of UsesCommonFeatures trait to solve the case.
        """
        from ...._settings import settings

        results = []
        for i, X in enumerate(embeddings):
            initial_dim = X.shape[1]
            X_sliced = self.slice_features(X, i)
            final_dim = X_sliced.shape[1]
            if (final_dim / initial_dim) < settings.low_features_ratio_threshold:
                self.warn(
                    "Low number of common features detected "
                    f"({initial_dim} features -> {final_dim} features, "
                    f"ratio < {settings.low_features_ratio_threshold}). "
                    "This can cause unexpected issues."
                )
            results.append(X_sliced)
        return results
