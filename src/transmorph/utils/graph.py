#!/usr/bin/env python3

import igraph as ig
import louvain as lv
import numpy as np
import warnings

from numba import njit
from numpy.random import RandomState
from pynndescent import NNDescent
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
from sklearn.neighbors import NearestNeighbors


from typing import Dict, List, Literal, Optional, Tuple

from transmorph.utils.dimred import pca


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
        Method to use ("auto", "exact" or "nndescent"). If "auto", will
        choose "exact" for small datasets and "nndescent" for large ones.

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
    if min(nx, ny) < n_neighbors:
        warnings.warn("Y.shape[0] < n_neighbors. " "Setting n_neighbors to Y.shape[0].")
        n_neighbors = min(nx, ny)
    D = cdist(X, Y, metric=metric, **metric_kwargs)
    dx = distance_to_knn(D, n_neighbors, 1)
    dy = distance_to_knn(D, n_neighbors, 0)
    Dxy = np.minimum.outer(dx, dy)
    return csr_matrix((D <= Dxy), shape=D.shape)


def nearest_neighbors(
    X: np.ndarray,
    mode: Literal["distances", "edges"] = "distances",
    algorithm: Literal["auto", "sklearn", "nndescent"] = "auto",
    n_neighbors: Optional[int] = None,
    metric: Optional[str] = None,
    metric_kwargs: Optional[Dict] = None,
    use_pcs: Optional[int] = None,
    include_self_loops: Optional[bool] = None,
    symmetrize: Optional[bool] = None,
    random_seed: Optional[int] = None,
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

    mode: Literal["distances", "edges"], default = "distances"
        Type of data contained in the returned matrix.

    use_pcs: int, default = 30
        If X.shape[1] > use_pcs, a PC view will be used instead of X.
        Set it to None to disable this functionality.

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
    # Retrieves default parameters if needed
    from .. import settings, use_setting

    n_neighbors = use_setting(n_neighbors, settings.n_neighbors)
    metric = use_setting(metric, settings.neighbors_metric)
    metric_kwargs = use_setting(metric_kwargs, settings.neighbors_metric_kwargs)
    use_pcs = use_setting(use_pcs, settings.neighbors_n_pcs)
    include_self_loops = use_setting(
        include_self_loops, settings.neighbors_include_self_loops
    )
    symmetrize = use_setting(symmetrize, settings.neighbors_symmetrize)
    random_seed = use_setting(random_seed, settings.neighbors_random_seed)
    # Checks parameters
    nx = X.shape[0]
    if algorithm == "nndescent":
        use_nndescent = True
    elif algorithm == "sklearn":
        use_nndescent = False
    elif algorithm == "auto":
        use_nndescent = nx > settings.large_dataset_threshold
    else:
        raise ValueError(
            f"Unrecognized algorithm: {algorithm}. Valid options are 'auto',"
            " 'nndescent', 'sklearn'."
        )
    assert mode in ("edges", "distances"), f"Unknown mode: {mode}."
    assert use_pcs is None or use_pcs > 0, f"Invalid PC number: {use_pcs}"

    if nx < n_neighbors:
        warnings.warn("X.shape[0] < n_neighbors. " "Setting n_neighbors to X.shape[0].")
        n_neighbors = nx

    if use_pcs is not None and use_pcs < X.shape[1]:
        X = pca(X, n_components=use_pcs)

    # Nearest neighbors
    connectivity = None
    if use_nndescent:
        # PyNNDescent provides a high speed implementation of kNN
        # Parameters borrowed from UMAP's implementation
        # https://github.com/lmcinnes/umap
        if not include_self_loops:
            n_neighbors += 1
        knn_result = NNDescent(
            X,
            n_neighbors=n_neighbors,
            metric=metric,
            metric_kwds=metric_kwargs,
            random_state=RandomState(random_seed),
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


def combine_matchings(
    knn_graphs: List[csr_matrix],
    matchings: Dict[Tuple[int, int], csr_matrix],
    mode: Literal["probability", "distance"],
    lam: float = 1.0,
) -> csr_matrix:
    """
    Concatenates any number of matchings Mij and knn-graph
    adjacency matrices Ai in a single sparse matrix T. Diagonal
    blocks of T is composed by Ai matrices, and ij block is
    Mij if i < j otherwise Mji.T

    matching: MatchingABC
        A fitted MatchingABC object with no reference.

    knn_graph: List[csr_matrix]
        List of knn-graphs, where knn-graph[i] is the knn-graph
        associated to matching.datasets[i].
    """
    rows, cols, data, N = [], [], [], 0
    offset_i = 0
    ndatasets = len(knn_graphs)
    for i in range(ndatasets):
        # Initial relations
        knn_graph = knn_graphs[i].tocoo()
        rows += list(knn_graph.row + offset_i)
        cols += list(knn_graph.col + offset_i)
        data += list(knn_graph.data)
        # Matchings
        offset_j = 0
        ni = knn_graphs[i].shape[0]
        for j in range(ndatasets):
            nj = knn_graphs[j].shape[0]
            if i >= j:
                offset_j += nj
                continue
            T = matchings[i, j]
            if T.shape[0] == nj:  # FIXME a lot of unnecessary conversions
                T = T.T
            if type(T) is csc_matrix or type(T) is csr_matrix:
                T = T.toarray()
            assert type(T) is np.ndarray, f"Unrecognized type: {type(T)}"
            norm = T.sum(axis=1, keepdims=True)
            norm[norm == 0.0] = 1.0
            T = csr_matrix(T / norm)
            matching_ij = T.tocoo()
            rows_k, cols_k = matching_ij.row, matching_ij.col
            rows_k += offset_i
            cols_k += offset_j
            rows += list(rows_k)  # Keep the symmetry
            rows += list(cols_k)
            cols += list(cols_k)
            cols += list(rows_k)
            data += 2 * len(cols_k) * [1]
            offset_j += nj
        offset_i += ni
        N += ni
    T = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    T.data = np.clip(T.data, 0.0, 1.0)
    T = T + T.T - T.multiply(T.T)  # symmetrize
    T.data[T.data == 0] = 1e-12  # Stabilize
    if mode == "probability":
        return T
    elif mode == "distance":
        # Gaussian model
        # pij = exp(-dij**2 * lambda)
        # iff dij = sqrt(-ln(pij) / lambda)
        # + epsilon to stabilize MDE solver
        T.data = np.sqrt(-np.log(T.data) / lam) + 1e-9
        return T
    else:
        raise ValueError(
            f"Mode {mode} unrecognized, should be 'probability'" " or 'distance'."
        )
