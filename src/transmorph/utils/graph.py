#!/usr/bin/env python3

import warnings
import numpy as np

from pynndescent import NNDescent
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.neighbors import NearestNeighbors

from numpy.random import RandomState

from typing import Union, Callable


def distance_to_knn(D, k, axis):
    """
    Returns the distance of each point along the specified axis to its kth
    nearest neighbor.
    """
    D_sorted = np.sort(D, axis=axis)
    if axis == 0:
        D_sorted = D_sorted.T
    return D_sorted[:, k]


def nearest_neighbors(
    X: np.ndarray,
    Y: np.ndarray = None,
    metric: Union[str, Callable] = "euclidean",
    metric_kwargs: dict = {},
    n_neighbors: int = 10,
    use_nndescent: bool = False,
    random_seed: int = 42,
    min_iters: int = 5,
    min_trees: int = 64,
    max_candidates: int = 60,
    low_memory: bool = False,
    n_jobs: int = -1,
) -> csr_matrix:
    """ """
    nx = X.shape[0]
    if nx < n_neighbors:
        warnings.warn("X.shape[0] < n_neighbors. " "Setting n_neighbors to X.shape[0].")
        n_neighbors = nx

    if Y is not None:  # Mutual nearest neighbors
        assert use_nndescent is False, "NNDescent is incompatible with MNN."
        ny = Y.shape[0]
        if ny < n_neighbors:
            warnings.warn(
                "Y.shape[0] < n_neighbors. " "Setting n_neighbors to Y.shape[0]."
            )
            n_neighbors = nx
        D = cdist(X, Y, metric=metric, **metric_kwargs)
        dx = distance_to_knn(D, n_neighbors, 1)
        dy = distance_to_knn(D, n_neighbors, 0)
        Dxy = np.minimum.outer(dx, dy)
        return csr_matrix(D[D <= Dxy])

    if use_nndescent:
        # PyNNDescent provides a high speed implementation of kNN
        # Parameters borrowed from UMAP's implementation
        # https://github.com/lmcinnes/umap
        n_trees = min(min_trees, 5 + int(round((X.shape[0]) ** 0.5 / 20.0)))
        n_iters = max(min_iters, int(round(np.log2(X.shape[0]))))
        knn_result = NNDescent(
            X,
            n_neighbors=n_neighbors,
            metric=metric,
            metric_kwds=metric_kwargs,
            random_state=RandomState(random_seed),
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=max_candidates,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=False,
        )
        knn_indices, knn_dists = knn_result.neighbor_graph
        rows, cols, data = [], [], []
        for i, (row_indices, row_dists) in enumerate(zip(knn_indices, knn_dists)):
            for j, dij in zip(row_indices, row_dists):
                rows.append(i)
                cols.append(j)
                data.append(dij)
        return coo_matrix((data, (rows, cols)), shape=(nx, nx)).tocsr()

    # Classical exact MNN
    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric,
        metric_params=metric_kwargs,
        n_jobs=n_jobs,
    )
    nn.fit(X)
    return nn.kneighbors_graph(X, n_neighbors=n_neighbors, mode="distance")
