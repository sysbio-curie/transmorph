#!/usr/bin/env python3

from ot.gromov import gromov_wasserstein

from scipy.spatial.distance import cdist


import numpy as np

from .matchingABC import MatchingABC


class MatchingGW(MatchingABC):
    """

    """
    def __init__(
            self,
            metric="sqeuclidean",
            metric_kwargs={},
            metric2=None,
            metric2_kwargs={},
            loss="square_loss",
            max_iter=1e6,
            use_sparse=True
    ):
        MatchingABC.__init__(self, use_sparse=use_sparse)
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        if metric2 is None:
            self.metric2 = self.metric
            self.metric2_kwargs = self.metric_kwargs
        else:
            self.metric2 = metric2
            self.metric2_kwargs = metric2_kwargs
        self.loss = loss
        self.max_iter = int(max_iter)


    def _match2(self, x1, x2):
        n1, n2 = x1.shape[0], x2.shape[0]
        w1, w2 = np.ones(n1) / n1, np.ones(n2) / n2
        M1 = cdist(
            x1,
            x1,
            metric=self.metric,
            **self.metric_kwargs
        )
        M1 /= M1.max()
        M2 = cdist(
            x2,
            x2,
            metric=self.metric2,
            **self.metric2_kwargs
        )
        M2 /= M2.max()
        return gromov_wasserstein(
            M1,
            M2,
            w1,
            w2,
            self.loss
        )