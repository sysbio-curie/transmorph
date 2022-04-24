#!/usr/bin/env python3

import numpy as np

from scipy.sparse import csr_matrix
from typing import List

from ..matching import Matching, _TypeMatchingSet
from ...traits.isprofilable import profile_method
from ...traits.usessamplelabels import UsesSampleLabels


class Labels(Matching, UsesSampleLabels):
    """
    This is a supervised boolean matching, that matches all
    samples sharing the same label.

    Parameters
    ----------
    label_obs: str
        AnnData obs key to extract label from. If left None,
        this trait is ignored.
    """

    def __init__(self, label_obs: str):
        Matching.__init__(self, str_identifier="LABELS")
        UsesSampleLabels.__init__(self, label_obs=label_obs)

    @profile_method
    def fit(self, datasets: List[np.ndarray]) -> _TypeMatchingSet:
        """
        Collects label matching matrices using UsesSampleLabels
        trait.
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
