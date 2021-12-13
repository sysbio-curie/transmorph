#!/usr/bin/env python3

from typing import Union, Callable

from .matchingABC import MatchingABC
from ..TData import TData
from ..utils import nearest_neighbors


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
        metric_kwargs: dict = {},
        n_neighbors: int = 10,
        use_sparse: bool = True,
    ):
        MatchingABC.__init__(self, use_sparse=use_sparse)
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.n_neighbors = n_neighbors

    def _match2(self, t1: TData, t2: TData):
        return nearest_neighbors(
            t1.X,
            Y=t2.X,
            metric=self.metric,
            metric_kwargs=self.metric_kwargs,
            n_neighbors=self.n_neighbors,
            use_nndescent=False,
        )
