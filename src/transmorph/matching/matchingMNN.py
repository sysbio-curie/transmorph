#!/usr/bin/env python3

from scipy.spatial.distance import cdist
from typing import (
    Union,
    Callable
)

import numpy as np

from .matchingABC import MatchingABC


class MatchingMNN(MatchingABC):
    """
    Mutual Nearest Neighbors (MNN) matching. Two samples xi and yj
    are matched if xi belongs to the k-nearest neighbors (kNNs) of yj
    and vice-versa. If we denote by dk(x) the distance from x to its
    kNN, then xi and yj are matched if d(xi, yj) < min{dk(xi), dk(yj)}.

    Parameters
    ----------
    metric: str or Callable, default = "sqeuclidean"
        Scipy-compatible metric.

    metric_kwargs: dict, default = {}
        Additional metric parameters.

    k: int, default = 10
        Number of neighbors to use for computing the kNNs.

    use_sparse: bool, default = True
        Save matching as sparse matrices.
    """
    def __init__(
            self,
            metric: Union[str, Callable] = "sqeuclidean",
            k: int = 10,
            use_sparse: bool = True
    ):
        MatchingABC.__init__(self, use_sparse=use_sparse)
        self.metric = metric
        self.k = k

    def _compute_di(self, D, axis):
        """
        Returns the distance of each point along the specified axis to its kth
        nearest neighbor.
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
