#!/usr/bin/env python3

import numpy as np

from scipy.sparse import csr_matrix
from typing import List

from ..matching import Matching, _TypeMatchingSet
from ...traits.isprofilable import profile_method
from ...traits.usessamplelabels import UsesSampleLabels


class Labels(Matching, UsesSampleLabels):
    """
    Supervised, label-based boolean matching that matches every pair
    of samples that share the same label. Does not depend on dataset
    dimensionality, and will typically generate a lot -- O(n^2) -- of
    edges.

    Parameters
    ----------
    label_obs: str
        AnnData.obs entry containing categorical labels.
    """

    def __init__(self, label_obs: str):
        Matching.__init__(self, str_identifier="LABELS")
        UsesSampleLabels.__init__(self, label_obs=label_obs)

    @profile_method
    def fit(
        self,
        datasets: List[np.ndarray],
    ) -> _TypeMatchingSet:
        """
        Collects label matching matrices using UsesSampleLabels
        trait, then computes the matching.
        """
        ndatasets = len(datasets)
        results: _TypeMatchingSet = {}
        for i in range(ndatasets):
            for j in range(ndatasets):
                if j >= i:
                    continue
                T = self.get_matching_matrix(i, j)
                results[i, j] = T
                results[j, i] = csr_matrix(T.T)
        return results
