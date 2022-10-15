#!/usr/bin/env python3

import anndata as ad
import numpy as np

from typing import List

from ..merging import Merging
from ...traits.usesreference import UsesReference


class Barycenter(Merging, UsesReference):
    """
    Barycentric merging is the most naive way of merging datasets.
    Given n vectorized datasets X1, ..., Xk, a reference dataset Y,
    and n row-normalized matchings T1, ..., Tk from Xis to Y, the
    barycentric merging f of Xi to Y through Ti is given by

        bary_Y(Xi, Ti) = Ti @ Y

    In other terms, every sample from Xi is projected to the weighted sum
    of its matches in Y. This merging is limited to complete matchings,
    where all points from Xi are matched to at least one point in Y, as
    unmatched points are naturally projected to the origin. In this case,
    LinearCorrection can circumvent this issue to a certain extent.
    """

    def __init__(self):
        Merging.__init__(
            self,
            preserves_space=False,
            str_identifier="BARYCENTER",
            matching_mode="normalized",
        )
        UsesReference.__init__(self)

    def transform(
        self, datasets: List[ad.AnnData], embeddings: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Combines datasets and matching to return a representation
        of all datasets in reference space.
        """
        k_ref = self.reference_index
        assert k_ref is not None, "Reference dataset must be set."
        X_ref = self.get_reference_item(embeddings)
        result = []
        for k, X in enumerate(embeddings):
            if X is X_ref:
                result.append(X_ref)
                continue
            T = self.get_matching(k, k_ref)
            result.append(T @ X_ref)
        return result
