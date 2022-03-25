#!/usr/bin/env python3

from ot.bregman import sinkhorn_stabilized
from scipy.spatial.distance import cdist

import numpy as np

from .matchingABC import MatchingABC
from ..subsampling.subsamplingABC import SubsamplingABC
from ..subsampling import SubsamplingKeepAll
import scanpy as sc


class MatchingSinkhorn(MatchingABC):
    """ """

    def __init__(
        self,
        metric="sqeuclidean",
        metric_kwargs={},
        epsilon=1e-2,
        max_iter=5e2,
        subsampling: SubsamplingABC = SubsamplingKeepAll(),
    ):
        super().__init__(metadata_keys=[], subsampling=subsampling)
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.epsilon = epsilon
        self.max_iter = int(max_iter)

    def _match2(self, adata1: sc.AnnData, adata2: sc.AnnData):
        X1 = self.to_match(adata1)
        X2 = self.to_match(adata2)
        n1, n2 = X1.shape[0], X2.shape[0]
        w1, w2 = np.ones(n1) / n1, np.ones(n2) / n2
        M = cdist(X1, X2, metric=self.metric, **self.metric_kwargs)
        M /= M.max()
        return sinkhorn_stabilized(w1, w2, M, self.epsilon, numItermax=self.max_iter)
