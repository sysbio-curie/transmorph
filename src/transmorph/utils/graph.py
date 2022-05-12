#!/usr/bin/env python3

import igraph as ig
import leidenalg as la
import numpy as np
import warnings

from anndata import AnnData
from numba import njit
from numpy.random import RandomState
from pynndescent import NNDescent
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Literal, Optional, Tuple

from .anndata_manager import anndata_manager as adm
from .dimred import pca
from .geometry import sparse_cdist
from .matrix import (
    extract_chunks,
    contains_duplicates,
    perturbate,
    sort_sparse_matrix,
    sparse_from_arrays,
)
from .._logging import logger


def cluster_anndatas(
    datasets: List[AnnData],
    use_rep: Optional[str] = None,
    cluster_key: str = "cluster",
    n_neighbors: int = 10,
    resolution: float = 1.0,
) -> None:
    """
    Runs Leiden algorithm on a set of concatenated anndatas objects embedded in a
    common space. Writes clustering results in the AnnData objects.

    Parameters
    ----------
    datasets: List[AnnData]
        AnnData objects to concatenante and cluster.

    use_rep: Optional[str], default = None
        Embedding to use, if None will use .obsm["transmorph"] in priority or raise
        an error if it is not found.

    cluster_key: str, default = "cluster"
        Key to save clusters in .obs

    n_neighbors: int, default = 10
        Number of neighbors to build the kNN graph.

    resolution: float, default = 1.0
        Leiden algorithm parameter.
    """
    if use_rep is None:
        use_rep = "transmorph"
    assert all(
        adm.isset_value(adata, key=use_rep, transmorph_key=False, field="obsm")
        for adata in datasets
    ), f"{use_rep} missing in .obsm of some AnnDatas."
    X = np.concatenate(
        [
            adm.get_value(adata, key=use_rep, transmorph_key=False, field="obsm")
            for adata in datasets
        ],
        axis=0,
    )
    adj_matrix = nearest_neighbors(X, mode="edges", n_neighbors=n_neighbors)
    sources, targets = adj_matrix.nonzero()
    edgelist = zip(sources.tolist(), targets.tolist())
    partition = np.array(
        la.find_partition(
            ig.Graph(edgelist),
            la.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
        ).membership
    )
    cluster_obs = extract_chunks(partition, [adata.n_obs for adata in datasets])
    for adata, obs in zip(datasets, cluster_obs):
        adm.set_value(adata, key=cluster_key, field="obs", value=obs, persist="output")


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


def qtree_mutual_nearest_neighbors(
    X: np.ndarray,
    Y: np.ndarray,
    qtX: NNDescent,
    qtY: NNDescent,
    n_neighbors: int = 10,
) -> csr_matrix:
    """
    Accelerated mutual nearest neighbors scheme using NNDescent.
    It requires precomputed indexes that can be queried.

    Parameters
    ----------
    X: np.ndarray
        First dataset, will be in rows in the final matrix

    Y: np.ndarray
        Second dataset, in columns in the final matrix

    qtX: NNDescent
        Precomputed index for X samples

    qtY: NNDescent
        Precomputed index for Y samples

    n_neighbors: int, default = 10
        Number of neighbors to use to build the intersection.
    """
    XYknn = sparse_from_arrays(
        qtY.query(X, k=n_neighbors)[0],
        n_cols=Y.shape[0],
    )
    YXknn = sparse_from_arrays(
        qtX.query(Y, k=n_neighbors)[0],
        n_cols=X.shape[0],
    )
    return XYknn.multiply(YXknn.T)


def raw_mutual_nearest_neighbors(
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

    n_neighbors = use_setting(n_neighbors, settings.n_neighbors_max)
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

    if use_pcs is not None and use_pcs < X.shape[1]:
        logger.debug(f"nearest_neighbors > Computing PCA {X.shape[1]} -> {use_pcs}")
        X = pca(X, n_components=use_pcs)

    # If overlapping points, adds a light noise to guarantee
    # NN algorithms proper functioning.
    if contains_duplicates(X):
        logger.debug("nearest_neighbors > Duplicates detected. Jittering.")
        X = perturbate(X, std=0.01)

    # By default, self loops are included in algorithms
    if not include_self_loops:
        n_neighbors += 1
        n_neighbors = min(n_neighbors, nx - 1)
    else:
        n_neighbors = min(n_neighbors, nx)

    # Nearest neighbors
    connectivity = None
    if use_nndescent:
        # PyNNDescent provides a high speed implementation of kNN
        # Parameters borrowed from UMAP's implementation
        # https://github.com/lmcinnes/umap
        logger.debug("nearest_neighbors > Computing nearest neighbors using nndescent.")
        q_tree = NNDescent(
            X,
            n_neighbors=n_neighbors,
            metric=metric,
            metric_kwds=metric_kwargs,
            random_state=RandomState(random_seed),
        )
        knn_indices, knn_distances = q_tree.neighbor_graph
        if not include_self_loops:  # Removes self-loops
            knn_indices = knn_indices[:, 1:]
            knn_distances = knn_distances[:, 1:]
        if mode == "distances":
            connectivity = sparse_from_arrays(knn_indices, knn_distances)
        else:
            connectivity = sparse_from_arrays(knn_indices)
    else:
        # Standard exact kNN using sklearn implementation
        logger.debug("nearest_neighbors > Computing nearest neighbors using sklearn.")

        nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=metric,
            metric_params=metric_kwargs,
        )
        nn.fit(X)
        if mode == "distances":
            nnmode = "distance"
        else:
            nnmode = "connectivity"
        connectivity = nn.kneighbors_graph(X, mode=nnmode)

        if not include_self_loops:
            connectivity.setdiag(0)
            connectivity.eliminate_zeros()

    if symmetrize:
        connectivity = fsymmetrize(connectivity)

    ks = (connectivity > 0).sum(axis=1)
    kmin, kmax = ks.min(), ks.max()
    logger.debug(f"nearest_neighbors > n: {X.shape[0]}, kmin: {kmin}, kmax: {kmax}")
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
    matchings: Dict[Tuple[int, int], csr_matrix],
    knn_graphs: List[csr_matrix],
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
    # Yuk
    ndatasets = max(max(matchings.keys(), key=lambda k: max(k))) + 1
    sizes = [X.shape[0] for X in knn_graphs]
    # Loading these containers
    rows, cols, data = [], [], []
    offset_i: int = 0
    for i in range(ndatasets):
        # Initial relations
        knn_graph = knn_graphs[i].tocoo()
        rows += list(knn_graph.row + offset_i)
        cols += list(knn_graph.col + offset_i)
        data += list(knn_graph.data)
        offset_j = offset_i + sizes[i]
        for j in range(i + 1, ndatasets):
            T = matchings[i, j].tocoo()
            rows_k, cols_k, data_k = T.row, T.col, T.data
            rows_k += offset_i
            cols_k += offset_j
            rows += list(rows_k)  # Keep the symmetry
            rows += list(cols_k)
            data += list(data_k)
            cols += list(cols_k)
            cols += list(rows_k)
            data += list(data_k)
            offset_j += sizes[j]
        offset_i += sizes[i]
    N = sum(sizes)
    T = csr_matrix((data, (rows, cols)), shape=(N, N))
    T.data = np.clip(T.data, 0.0, 1.0)
    T = T + T.T - T.multiply(T.T)  # symmetrize
    return T


def generate_membership_matrix(
    G: csr_matrix,
    X1: np.ndarray,
    X2: np.ndarray,
    max_iter: int = 64,
    tol: float = 1e-5,
    low_thr: float = 1e-6,
) -> csr_matrix:
    """
    Converts a graph matrix G between two datasets X1 and X2 embedded in the same
    space to a distance-based membership matrix, ready to be embedded by UMAP or
    MDE optimizers. Adapted from umap-learn package.
    """
    # Sanity checks
    assert G.shape == (X1.shape[0], X2.shape[0])

    # Retrieving distances is possible, otherwise guessing them
    if X1.shape[1] != X2.shape[1]:  # Not same space
        G_dist = G / G.max()
        G_dist.data = 1.0 / (1.0 + G_dist.data)  # FIXME?
    else:  # Same space
        G_dist = sparse_cdist(X1, X2, G, metric="euclidean")

    # Initialization
    indices, distances = sort_sparse_matrix(G_dist, fill_empty=True)
    distances = _generate_membership_matrix_njit(
        distances,
        max_iter,
        tol,
    )
    membership = sparse_from_arrays(indices, distances, n_cols=X2.shape[0])
    assert membership.max() != np.inf, "np.inf deteceted in $membership."
    membership.data[membership.data < low_thr] = 0.0
    membership.eliminate_zeros()
    return membership


@njit(fastmath=True)
def _generate_membership_matrix_njit(
    distances: np.ndarray,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """
    Numba accelerated helper
    """
    n1 = distances.shape[0]
    # Binary search
    for row_i in range(n1):

        # Skip the line if empty
        if distances[row_i, 0] == np.inf:
            continue

        low = 0.0
        mid = 1.0
        high = np.inf

        rhos_i = distances[row_i, 0]
        k = 0
        for r in distances[row_i]:
            if r != np.inf:
                k += 1
        target_log2k = np.log2(k)

        for _ in range(max_iter):

            cand_log2k = 0.0
            for j in range(k):

                # End of neighbors
                if distances[row_i, j] == np.inf:
                    continue

                d = distances[row_i, j] - rhos_i
                if d <= 0:
                    cand_log2k += 1
                else:
                    cand_log2k += np.exp(-(d / mid))

            if np.abs(cand_log2k - target_log2k) < tol:
                break

            if cand_log2k > target_log2k:
                high = mid
                mid = (low + high) / 2.0
            else:
                low = mid
                if high is np.inf:
                    mid *= 2.0
                else:
                    mid = (low + high) / 2.0

        distances[row_i] = np.exp(-np.clip(distances[row_i] - rhos_i, 0, None) / mid)

    return distances


def get_nearest_vertex_from_set(
    indices: np.ndarray,
    distances: np.ndarray,
    vset: np.ndarray,
) -> np.ndarray:
    """
    Returns for each vertex the closest vertex from vset along graph G edges.
    If the connected component of a vertex contains no vertex from vset,
    its nearest vertex from vset if set to -1.

    Parameters
    ----------
    indices, distances: np.ndarray
        Edge/Distance matrices of shape (n, k) representing the knn graph.

    vset: np.ndarray
        Boolean vector representing vertices of the target set.

    TODO: This could be done more intelligenlty without redundant pathing.
          It will do the job for now.
    """
    nvertices = indices.shape[0]
    # Default is -1
    result = np.zeros((nvertices,), dtype=int) - 1
    for i in range(nvertices):
        # List of (idx, dist to vi)
        to_visit = [(i, 0.0)]
        visited = set([i])
        while len(to_visit) > 0:
            current_v, dv = to_visit.pop(0)
            # Vertex from vset is found
            if vset[current_v]:
                result[i] = current_v
                break
            # Adds unvisited vertices to to_visit, sorted
            for nb, dnb in zip(indices[current_v], distances[current_v]):
                if nb == -1 or nb in visited:
                    continue
                visited.add(nb)
                dnb += dv
                inserted = False
                for k, (_, tot_d) in enumerate(to_visit):
                    if tot_d >= dnb:
                        inserted = True
                        to_visit.insert(k, (nb, dnb))
                        break
                if not inserted:
                    to_visit.append((nb, dnb))
    return result
