#!/usr/bin/env python3

from ot import emd
from scipy.spatial.distance import cdist

from .matchingABC import MatchingABC

import numpy as np


class MatchingEMD(MatchingABC):
    """

    """
    def __init__(
            self,
            metric="sqeuclidean",
            metric_kwargs={},
            max_iter=1e6,
            use_sparse=True
    ):
        MatchingABC.__init__(self, use_sparse=use_sparse)
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.max_iter = int(max_iter)


    def _match2(self, x1, x2):
        n1, n2 = x1.shape[0], x2.shape[0]
        w1, w2 = np.ones(n1) / n1, np.ones(n2) / n2
        M = cdist(
            x1,
            x2,
            metric=self.metric,
            **self.metric_kwargs
        )
        M /= M.max()
        return emd(w1, w2, M, numItermax=self.max_iter)