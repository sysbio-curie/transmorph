#!/usr/bin/env python3

from ot.bregman import sinkhorn_stabilized
from scipy.spatial.distance import cdist

import numpy as np

from .matchingABC import MatchingABC
from ..TData import TData


class MatchingSinkhorn(MatchingABC):
    """ """

    def __init__(
        self,
        metric="sqeuclidean",
        metric_kwargs={},
        epsilon=1e-2,
        max_iter=5e2,
        use_sparse=True,
    ):
        MatchingABC.__init__(self, use_sparse=use_sparse)
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.epsilon = epsilon
        self.max_iter = int(max_iter)

    def _match2(self, t1: TData, t2: TData):
        n1, n2 = t1.X.shape[0], t2.X.shape[0]
        w1, w2 = np.ones(n1) / n1, np.ones(n2) / n2
        M = cdist(t1.X, t2.X, metric=self.metric, **self.metric_kwargs)
        M /= M.max()
        return sinkhorn_stabilized(w1, w2, M, self.epsilon, numItermax=self.max_iter)
