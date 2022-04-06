#!/usr/bin/env python3

from anndata import AnnData
from scipy.sparse import csr_matrix
from typing import Dict, Optional

from .matchingABC import MatchingABC
from ..subsampling.subsamplingABC import SubsamplingABC
from ..utils import mutual_nearest_neighbors


class MatchingMNN(MatchingABC):
    """
    Mutual Nearest Neighbors (MNN) matching. Two samples xi and yj
    are matched if xi belongs to the k-nearest neighbors (kNNs) of yj
    and vice-versa. If we denote by dk(x) the distance from x to its
    kNN, then xi and yj are matched if d(xi, yj) < min{dk(xi), dk(yj)}.
    In other terms,
    x \\in X and y \\in Y are mutual nearest neighbors if
    - y belongs to the $k nearest neighbors of x in Y
    AND
    - x belongs to the $k nearest neighbors of y in X

    You can choose between two methods:
    - The exact MNN solver, with high fiability but which can become
      computationally prohibitive when datasets scale over tens of
      thousands of samples.
    - An experimental approached solver, which matches samples between
      matched clusters, less fiable but more tractable for large problems.
      This solver will be subject to improvements.

    Parameters
    ----------
    metric: str or Callable, default = "sqeuclidean"
        Scipy-compatible metric.

    metric_kwargs: dict, default = {}
        Additional metric parameters.

    n_neighbors: int, default = 10
        Number of neighbors to use between datasets.
        "features" key, a list of features names.

    algorithm: str, default = "auto"
        Method to use ("auto", "exact" or "louvain"). If "auto", will
        choose "exact" for small datasets and "louvain" for large ones.

    subsampling: SubsamplingABC, default = None
        Subsampling scheme to apply before computing the matching,
        can be very helpful when dealing with large datasets.
    """

    def __init__(
        self,
        metric: str = "sqeuclidean",
        metric_kwargs: Optional[Dict] = None,
        n_neighbors: int = 10,
        algorithm="auto",
        subsampling: Optional[SubsamplingABC] = None,
    ):
        super().__init__(metadata_keys=[], subsampling=subsampling)
        self.metric = metric
        self.metric_kwargs = {} if metric_kwargs is None else metric_kwargs
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm

    def _match2(self, adata1: AnnData, adata2: AnnData) -> csr_matrix:
        X = MatchingABC.to_match(adata1)
        Y = MatchingABC.to_match(adata2)
        T = mutual_nearest_neighbors(
            X,
            Y,
            metric=self.metric,
            metric_kwargs=self.metric_kwargs,
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm,
        )
        return T
