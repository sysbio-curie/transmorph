#!/usr/bin/env python3

import warnings
import numpy as np

from numba import njit
from numpy.random import RandomState
from pynndescent import NNDescent
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.neighbors import NearestNeighbors


from typing import Tuple, Union, Callable


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
    metric: Union[str, Callable] = "sqeuclidean",
    metric_kwargs: dict = {},
    n_neighbors: int = 10,
    include_self_loops: bool = False,
    symmetrize: bool = False,
    use_nndescent: bool = False,
    random_seed: int = 42,
    min_iters: int = 5,
    min_trees: int = 64,
    max_candidates: int = 60,
    low_memory: bool = False,
    n_jobs: int = -1,
) -> csr_matrix:
    """
    Encapsulates both Nearest neighbors and Mutual nearest neighbors computations.

    Parameters
    ----------
    X: np.ndarray
        Vectorized dataset to compute nearest neighbors from.

    Y: np.ndarray, default = None
        Vectorized reference dataset to use in the mutual nearest neighbors case.
        For nearest neighbors, leave this as None.

    metric: str or Callable, default = "sqeuclidean"
        scipy-compatible metric used to compute nearest neighbors.

    metric_kwargs: dict, default = {}
        Additional metric arguments for scipy.spatial.distance.cdist.

    include_self_loops: bool, default = False
        Whether points are neighbors of themselves.

    symmetrize: bool, default = False
        Make edges undirected

    use_nndescent: bool, default = False
        Use a nearest neighbors descent heuristic for higher computational
        efficiency. Not compatible with mutual nearest neighbors. We use
        the pynndescent implementation. All next parameters are pynndescent
        parameters, refer to their documentation for more info.
    """
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
        return csr_matrix((D <= Dxy), shape=D.shape)

    # Nearest neighbors
    connectivity = None
    if use_nndescent:
        # PyNNDescent provides a high speed implementation of kNN
        # Parameters borrowed from UMAP's implementation
        # https://github.com/lmcinnes/umap
        if not include_self_loops:
            n_neighbors += 1
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
        knn_indices, _ = knn_result.neighbor_graph
        rows, cols, data = [], [], []
        for i, row_indices in enumerate(knn_indices):
            for j in row_indices:
                if not include_self_loops and i == j:
                    continue
                rows.append(i)
                cols.append(j)
                data.append(1.0)
        connectivity = coo_matrix((data, (rows, cols)), shape=(nx, nx)).tocsr()

    else:

        # Classical exact MNN
        nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=metric,
            metric_params=metric_kwargs,
            n_jobs=n_jobs,
        )
        nn.fit(X)
        if include_self_loops:
            connectivity = nn.kneighbors_graph(X, n_neighbors=n_neighbors)
        else:
            connectivity = nn.kneighbors_graph(n_neighbors=n_neighbors)

    if symmetrize:
        connectivity = (
            connectivity + connectivity.T - connectivity.multiply(connectivity.T)
        )
    return connectivity


def vertex_cover(adjacency: csr_matrix, hops: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes a vertex cover S of a graph G = (V, E) defined by its adjacency
    matrix. S has a fundamental property: for all v in V, either v in S or
    there exists (v, x) in E so that x in V. S is a subset of V that can
    then be used as a subsampling for heavy computational tasks. The heuristic
    we implemented is a O(n) in time, and returns a vertex cover at most twice
    as large as the smallest one.

    Parameters
    ----------
    adjacency: csr_matrix
        Adjacency matrix of the graph to subsample, must be symmetrical.

    hops: int, default = 1
        Maximal distance to a selected node to be considered covered.

    Returns
    -------
    (cover, mapping): Tuple[np.ndarray, np.ndarray]
        cover: Boolean vector representing belonging to the cover
        mapping: mapping[i] is the index of the node covering node i.
    """
    assert hops > 0, "Hops must be positive."
    if hops == 0:
        n = adjacency.shape[0]
        cover, mapping = np.ones(n), np.arange(n)
    else:  # use a numba-accelerated function
        cover, mapping = vertex_cover_njit(
            adjacency.indptr, adjacency.indices, hops=hops
        )
    return cover.astype(bool), mapping.astype(int)


@njit
def vertex_cover_njit(
    ptr: np.ndarray, ind: np.ndarray, hops: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    # ptr, ind: CSR matrix representation
    # of an adjacency matrix
    n = ptr.shape[0] - 1
    anchors = np.zeros(n, dtype="int")
    for v in range(n):
        # If v not visited, add it as an anchor
        if anchors[v]:
            continue
        anchors[v] = v
        # Mark its neighbors as visited
        neighbors = [(v, 0)]
        while len(neighbors):
            nb, d = neighbors.pop()
            anchors[nb] = v
            if d < hops:
                M = ptr[nb + 1] if nb + 1 < n else n
                for nb2 in ind[ptr[nb] : M]:
                    if anchors[nb2]:
                        continue
                    anchors[nb2] = v
                    neighbors.append((nb2, d + 1))
    anchors_set = np.zeros(n)
    for i in set(anchors):
        anchors_set[i] = 1
    return anchors_set, anchors  # set, map
