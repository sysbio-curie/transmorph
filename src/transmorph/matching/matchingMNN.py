#!/usr/bin/env python3

from scipy.spatial.distance import cdist

import numpy as np

from .matchingABC import MatchingABC


class MatchingMNN(MatchingABC):
    """

    """
    def __init__(
            self,
            metric="sqeuclidean",
            k=10,
            use_sparse=True
    ):
        MatchingABC.__init__(self, use_sparse=use_sparse)
        self.metric = metric
        self.k = k

    def _compute_di(self, D, axis):
        """
        Returns the distance of each xi to its kth nearest neighbor
        """
        D_sorted = np.sort(D, axis=axis)
        if axis == 0:
            D_sorted = D_sorted.T
        return D_sorted[:, self.k]

    def _match2(self, x1, x2):
        D = cdist(x1, x2, metric=self.metric)
        dx = self._compute_di(D, axis=1)
        dy = self._compute_di(D, axis=0)
        Dxy = np.minimum.outer(dx, dy)
        return D <= Dxy