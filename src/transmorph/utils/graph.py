#!/usr/bin/env python3

import igraph as ig
import louvain as lv
import numpy as np
import warnings

from numba import njit
from numpy.random import RandomState
from pynndescent import NNDescent
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.neighbors import NearestNeighbors


from typing import Dict, Optional, Tuple


def fsymmetrize(A: csr_matrix) -> csr_matrix:
    # Symmetrizes a probabilistic matching matrix, given
    # the rule P(AUB) = P(A) + P(B) - P(A)P(B), assuming
    # A and B independent events.
    return A + A.T - A.multiply(A.T)


def distance_to_knn(D: np.ndarray, k: int, axis: int):
    """
    Returns the distance of each point along the specified axis to its kth
    nearest neighbor.
    """
    D_sorted = np.sort(D, axis=axis)
    if axis == 0:
        D_sorted = D_sorted.T
    return D_sorted[:, k - 1]


def clustering(A: csr_matrix, resolution: float = 1.0) -> np.ndarray:
    """
    Uses Louvain algorithm to provide a clustering of the directed
    unweighted graph represented as matrix A.

    Parameters
    ----------
    A: csr_matrix
        Adjacency matrix of shape (n,n).

    resolution: float, default = 1.0
        Resolution parameter for Louvain algorithm.
        #TODO: adaptive selection of this parameter?

    Returns
    -------
    (n,) np.ndarray containing cluster affectation as integers
    """
    sources, targets = A.nonzero()
    A_ig = ig.Graph(directed=True)
    A_ig.add_vertices(A.shape[0])
    A_ig.add_edges(list(zip(sources, targets)))
    partition = lv.find_partition(
        A_ig, lv.RBConfigurationVertexPartition, resolution_parameter=resolution
    )
    return np.array(partition.membership)


def mutual_nearest_neighbors(
    X: np.ndarray,
    Y: np.ndarray,
    metric: str = "sqeuclidean",
    metric_kwargs: Dict = {},
    n_neighbors: int = 10,
    algorithm: str = "auto",
    low_memory: bool = False,
    n_jobs: int = -1,
) -> csr_matrix:
    """
    Runs mutual nearest neighbors algorithm between datasets X and Y.
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
    X: np.ndarray
        First dataset of shape (n,d)

    Y: np.ndarray
        Second dataset of shape (m,d)

    metric: str, default = "sqeuclidean"
        scipy-compatible metric to use.

    metric_kwargs: Dict, default = {}
        Additional parameters for metric

    n_neighbors: int, default = 10
        Number of neighbors to use between datasets.

    algorithm: str, default = "auto"
        Method to use ("auto", "exact" or "louvain"). If "auto", will
        choose "exact" for small datasets and "louvain" for large ones.

    low_memory: bool, default = False
        Run pynndescent using high time/low memory profile for large
        datasets. Turn it on for very large datasets where memory is an
        issue.

    n_jobs: int = -1
        Number of jobs to pass to sklearn nearest_neighbors function.

    Returns
    -------
    T = (n,m) csr_matrix where T[i,j] = (xi and yj MNN)
    """
    nx, ny = X.shape[0], Y.shape[0]
    npoints = nx + ny
    if algorithm == "auto":
        algorithm = "louvain" if npoints > 4096 else "exact"
    if algorithm == "louvain" and npoints < 500:
        algorithm = "exact"
    if algorithm == "exact":
        if min(nx, ny) < n_neighbors:
            warnings.warn(
                "Y.shape[0] < n_neighbors. " "Setting n_neighbors to Y.shape[0]."
            )
            n_neighbors = min(nx, ny)
        D = cdist(X, Y, metric=metric, **metric_kwargs)
        dx = distance_to_knn(D, n_neighbors, 1)
        dy = distance_to_knn(D, n_neighbors, 0)
        Dxy = np.minimum.outer(dx, dy)
        return csr_matrix((D <= Dxy), shape=D.shape)
    if algorithm == "louvain":
        # Computing approached kNN matrices of X and Y
        Ax = nearest_neighbors(
            X,
            metric=metric,
            metric_kwargs=metric_kwargs,
            n_neighbors=n_neighbors,
            algorithm="nndescent",
            low_memory=low_memory,
            n_jobs=n_jobs,
        )
        Ay = nearest_neighbors(
            Y,
            metric=metric,
            metric_kwargs=metric_kwargs,
            n_neighbors=n_neighbors,
            algorithm="nndescent",
            low_memory=low_memory,
            n_jobs=n_jobs,
        )
        # Clustering Ax and Ay
        px, py = clustering(Ax), clustering(Ay)
        ncx, ncy = len(set(px)), len(set(py))
        npart = max(ncx, ncy)
        centroidx = np.array([np.mean(X[px == k], axis=0) for k in range(ncx)])
        centroidy = np.array([np.mean(Y[py == k], axis=0) for k in range(ncy)])
        # MNN matching of cluster centroids
        part_matching = mutual_nearest_neighbors(
            centroidx,
            centroidy,
            metric=metric,
            metric_kwargs=metric_kwargs,
            n_neighbors=npart // 3,  # Guess
            algorithm="exact",
        ).toarray()
        # Match points only between matched clusters
        rows, cols = [], []
        for i in range(ncx):
            indices_i = np.arange(nx)[px == i]
            for j in range(ncy):
                indices_j = np.arange(ny)[py == j]
                if part_matching[i, j]:
                    Tij = mutual_nearest_neighbors(
                        X[px == i],
                        Y[py == j],
                        metric=metric,
                        metric_kwargs=metric_kwargs,
                        n_neighbors=n_neighbors,  # Guess :D
                        algorithm="auto",
                        low_memory=low_memory,
                        n_jobs=n_jobs,
                    ).tocoo()
                    rows += list(indices_i[Tij.row])
                    cols += list(indices_j[Tij.col])
        data = [1] * len(rows)
        return coo_matrix((data, (rows, cols)), shape=(nx, ny)).tocsr()

    raise ValueError(f"Unknown algorithm: {algorithm}")


def nearest_neighbors(
    X: np.ndarray,
    metric: str = "sqeuclidean",
    metric_kwargs: dict = {},
    n_neighbors: int = 10,
    include_self_loops: bool = False,
    symmetrize: bool = False,
    algorithm: str = "auto",
    random_seed: int = 42,
    min_iters: int = 5,
    min_trees: int = 64,
    max_candidates: int = 60,
    low_memory: bool = False,
    n_jobs: int = -1,
    use_nndescent: Optional[bool] = None,
) -> csr_matrix:
    """
    Encapsulates k-nearest neighbors algorithms.

    Parameters
    ----------
    X: np.ndarray
        Vectorized dataset to compute nearest neighbors from.

    metric: str or Callable, default = "sqeuclidean"
        scipy-compatible metric used to compute nearest neighbors.

    metric_kwargs: dict, default = {}
        Additional metric arguments for scipy.spatial.distance.cdist.

    n_neighbors: int, default = 10
        Number of neighbors to use between datasets.

    include_self_loops: bool, default = False
        Whether points are neighbors of themselves.

    symmetrize: bool, default = False
        Make edges undirected

    algorithm: str, default = "auto"
        Solver to use in "auto", "sklearn" and "nndescent". Use "sklearn"
        for small datasets or if the solution must be exact, use "nndescent"
        for large datasets if an approached solution is enough. With "auto",
        the function will adapt to dataset size.

    """
    nx = X.shape[0]
    if use_nndescent is not None:
        warnings.warn(
            "use_nndescent is deprecated and will be removed in the future. "
            "Please use 'algorithm' instead."
        )
    elif algorithm == "nndescent":
        use_nndescent = True
    elif algorithm == "sklearn":
        use_nndescent = False
    elif algorithm == "auto":
        use_nndescent = nx > 4096
    else:
        raise ValueError(
            f"Unrecognized algorithm: {algorithm}. Valid options are 'auto',"
            " 'nndescent', 'sklearn'."
        )

    if nx < n_neighbors:
        warnings.warn("X.shape[0] < n_neighbors. " "Setting n_neighbors to X.shape[0].")
        n_neighbors = nx

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
        # Standard exact kNN using sklearn implementation
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
        connectivity = fsymmetrize(connectivity)
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
    assert hops >= 0, "Hops must be non-negative."
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
