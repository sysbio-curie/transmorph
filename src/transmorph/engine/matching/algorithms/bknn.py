#!/usr/bin/env python3

import numpy as np

from scipy.sparse import csr_matrix
from typing import Dict, List, Literal, Optional

from ..matching import Matching, _TypeMatchingSet
from ...traits.isprofilable import profile_method
from ...traits.usescommonfeatures import UsesCommonFeatures
from ...traits.usesreference import UsesReference
from ....utils.graph import (
    generate_qtree,
    qtree_k_nearest_neighbors,
    nearest_neighbors_custom,
)


class BKNN(Matching, UsesCommonFeatures, UsesReference):
    """
    Batch k-nearest neighbors matching. x from batch X is matched with
    y from batch Y if y belongs to the k nearest neighbors of x in Y.

    We propose two solvers,

    - An exact solver, which computes all pairwise distances between
      batches, computes exact nearest neighbors and makes the intersection
    - An approximate solver, that leverages NNDescent algorithm to
      compute approximate nearest neighbors using projection trees.
      This allows MNN to scale to large, high dimensional datasets.

    Parameters
    ----------
    metric: str or Callable, default = "sqeuclidean"
        Scipy-compatible metric for kNN computation.

    metric_kwargs: dict, default = {}
        Additional metric parameters.

    n_neighbors: int, default = 10
        Number of neighbors to use between batches.

    common_features_mode: Literal["pairwise", "total"], default = "pairwise"
        Uses pairwise common features, or total common features. Use "total"
        for a small number of datasets, and "pairwise" if the features
        intersection is too small. Ignored if datasets are not in feature
        space, set to "total" if solver = "pynndescent".

    solver: Literal["auto", "exact", "pynndescent"], default = "auto"
        Solver to use.
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
        UsesReference.__init__(self)
        self.metric = metric
        self.metric_kwargs = {} if metric_kwargs is None else metric_kwargs
        self.n_neighbors = n_neighbors

    @profile_method
    def fit(
        self,
        datasets: List[np.ndarray],
    ) -> _TypeMatchingSet:
        """
        Computes BKNN between pairs of datasets.
        """
        ndatasets = len(datasets)
        results: _TypeMatchingSet = {}
        reference = self.reference_index
        if reference is None:
            target_indices = np.arange(ndatasets)
        else:
            target_indices = [reference]
        small_scale_problem = all(X.shape[0] < 100 for X in datasets)
        if not small_scale_problem:
            qtrees = [
                generate_qtree(X, self.metric, self.metric_kwargs) for X in datasets
            ]
        for i in range(ndatasets):
            for j in target_indices:
                if i == j:
                    continue
                Xi, Yi = self.slice_features(
                    X1=datasets[i],
                    X2=datasets[j],
                    idx_1=i,
                    idx_2=j,
                )
                if small_scale_problem:
                    Tij = nearest_neighbors_custom(
                        Xi, mode="edges", n_neighbors=self.n_neighbors, Y=Yi
                    )
                else:
                    Tij = qtree_k_nearest_neighbors(
                        Xi,
                        qtrees[j],
                        n_neighbors=self.n_neighbors,
                    )
                results[i, j] = csr_matrix(Tij)
        return results
