#!/usr/bin/env python3

import numpy as np

from typing import List

from .. import Merging
from ...traits import UsesReference


class Barycenter(Merging, UsesReference):
    """
    Barycentric merging is the most naive way of merging datasets.
    Given n vectorized datasets x1, ..., xn and a reference dataset y,
    and n row-normalized matchings m1, ..., mn, we compute the following
    projection:

        f(xi) = mi @ y

    In other terms, every sample from xi is projected to the weighted sum
    of its matches in y. This works quite well with optimal transport-based
    methods, but is quite limited in the MNN case due to unmatched samples.
    Fitted, referenced matching between datasets.
    """

    def __init__(self):
        Merging.__init__(
            self,
            preserves_space=False,
            str_identifier="BARYCENTER",
            matching_mode="normalized",
        )

    def transform(self, datasets: List[np.ndarray]) -> List[np.ndarray]:
        """
        Combines datasets and matching to return a representation
        of all datasets in reference space.
        """
        k_ref = self.reference_index
        assert k_ref is not None, "Reference dataset must be set."
        X_ref = self.get_reference_index(datasets)
        result = []
        for k, X in enumerate(datasets):
            if X == X_ref:
                result.append(X_ref)
                continue
            T = self.get_matching(k, k_ref)
            result.append(T @ X)
        return result
