#!/usr/bin/env python3

import numpy as np

from scipy.sparse import csr_matrix
from typing import Dict, List, Literal, Optional

from ..matching import Matching, _TypeMatchingSet
from ...traits.isprofilable import profile_method
from ...traits.usescommonfeatures import UsesCommonFeatures
from ....utils import mutual_nearest_neighbors


class MatchingMNN(Matching, UsesCommonFeatures):
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
        common_features_mode: Literal["pairwise", "total"] = "pairwise",
    ):
        Matching.__init__(self, str_identifier="MNN")
        UsesCommonFeatures.__init__(self, mode=common_features_mode)
        self.metric = metric
        self.metric_kwargs = {} if metric_kwargs is None else metric_kwargs
        self.n_neighbors = n_neighbors

    @profile_method
    def fit(self, datasets: List[np.ndarray]) -> _TypeMatchingSet:
        """
        Computes MNN between pairs of datasets.
        """
        ndatasets = len(datasets)
        results: _TypeMatchingSet = {}
        for i in range(ndatasets):
            for j in range(i + 1, ndatasets):
                Xi, Xj = datasets[i], datasets[j]
                Xj, Xj = self.slice_features(X1=Xi, X2=Xj, idx_1=i, idx_2=j)
                Tij = mutual_nearest_neighbors(
                    Xi,
                    Xj,
                    metric=self.metric,
                    metric_kwargs=self.metric_kwargs,
                    n_neighbors=self.n_neighbors,
                )
                results[i, j] = csr_matrix(Tij)
                results[j, i] = csr_matrix(Tij.T)
        return results
