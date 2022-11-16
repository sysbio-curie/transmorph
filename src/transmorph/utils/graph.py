#!/usr/bin/env python3

import anndata as ad
import igraph as ig
import leidenalg as la
import numpy as np
import scanpy as sc
import warnings

from anndata import AnnData
from numba import njit
from numpy.random import RandomState
from pynndescent import NNDescent
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn import neighbors as skn
from typing import Dict, List, Literal, Optional, Tuple

from .anndata_manager import anndata_manager as adm
from .geometry import sparse_cdist
from .matrix import (
    extract_chunks,
    sort_sparse_matrix,
    sparse_from_arrays,
)


def nearest_neighbors_custom(
    X: np.ndarray,
    mode: Literal["edges", "distances"],
    n_neighbors: int = 15,
    Y: Optional[np.ndarray] = None,
) -> csr_matrix:
    """
    Wraps sklearn.neighbors.NearestNeighbors class.
    """
    if Y is None:
        Y = X
    nn_model = skn.NearestNeighbors(n_neighbors=n_neighbors).fit(Y)
    if mode == "edges":
        return nn_model.kneighbors_graph(X, mode="connectivity")
    elif mode == "distances":
        return nn_model.kneighbors_graph(X, mode="distance").toarray()
    else:
        raise ValueError(f"Unrecognized mode: {mode}")


def nearest_neighbors(
    adata: ad.AnnData,
    mode: Literal["edges", "distances", "connectivities"],
    neighbors_key: str = "neighbors",
    n_neighbors: int = 15,
    metric: str = "euclidean",
    metric_kwargs: Dict = {},
    use_pca: bool = True,
    random_state: Optional[RandomState] = None,
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

    algorithm: str, default = "auto"
        Solver to use in "auto", "sklearn" and "nndescent". Use "sklearn"
        for small datasets or if the solution must be exact, use "nndescent"
        for large datasets if an approached solution is enough. With "auto",
        the function will adapt to dataset size.
    """
    assert isinstance(
        adata, ad.AnnData
    ), f"AnnData expected, found {type(adata).__name__}."
    assert mode in ["edges", "distances", "connectivities"]
    if (
        neighbors_key not in adata.uns
        or "params" not in adata.uns[neighbors_key]
        or "n_neighbors" not in adata.uns[neighbors_key]["params"]
        or "metric" not in adata.uns[neighbors_key]["params"]
        or adata.uns[neighbors_key]["params"]["n_neighbors"] < n_neighbors
        or adata.uns[neighbors_key]["params"]["metric"] != metric
    ):
        from .._settings import settings, use_setting

        if use_pca and "X_pca" not in adata.obsm:
            sc.pp.pca(adata)
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            metric=metric,
            metric_kwds=metric_kwargs,
            method="umap",
            random_state=use_setting(random_state, settings.global_random_seed),
        )
    if mode == "edges":
        return adata.obsp["distances"].astype(bool)
    elif mode == "distances":
        return adata.obsp["distances"]
    elif mode == "connectivities":
        return adata.obsp["connectivities"]


def distance_to_knn(D: np.ndarray, k: int, axis: int):
    """
    Returns the distance of each point along the specified axis to its kth
    nearest neighbor, given a distance matrix D.
    """
    if k == 0:  # 0-th neighbor of a point is itself
        return np.zeros(D.shape[1 - axis], dtype=np.float32)
    D_sorted = np.sort(D, axis=axis)
    if axis == 0:
        D_sorted = D_sorted.T
    return D_sorted[:, k - 1]


def generate_qtree(
    X: np.ndarray, metric: str, metric_kwargs: Optional[Dict] = None
) -> NNDescent:
    """
    Returns a fitted tree based on points from X, which can be
    then queried for another set of points.
    """
    if metric_kwargs is None:
        metric_kwargs = {}
    qtree = NNDescent(X, metric=metric, metric_kwds=metric_kwargs)
    qtree.prepare()
    return qtree


def qtree_k_nearest_neighbors(
    X: np.ndarray,
    qtY: NNDescent,
    n_neighbors: int = 10,
) -> csr_matrix:
    """
    Returns k nearest neighbors between X and Y using a k-d tree
    algorithm, as a csr_matrix.

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
    return sparse_from_arrays(
        qtY.query(X, k=n_neighbors)[0],
        n_cols=qtY._raw_data.shape[0],
    )


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
    XYknn = qtree_k_nearest_neighbors(X, qtY, n_neighbors)
    YXknn = qtree_k_nearest_neighbors(Y, qtX, n_neighbors)
    return XYknn.multiply(YXknn.T).astype(np.float32)


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
    return csr_matrix((D <= Dxy), shape=D.shape, dtype=np.float32)


def combine_matchings(
    matchings: Dict[Tuple[int, int], csr_matrix],
    knn_graphs: Optional[List[csr_matrix]] = None,
    symmetrize: bool = True,
    target_edges: int = 10,
) -> csr_matrix:
    """
    Concatenates any number of matchings Mij and knn-graph
    adjacency matrices Ai in a single sparse matrix T. Diagonal
    blocks of T is composed by Ai matrices, and ij block is
    Mij if i < j otherwise Mji.T

    matching: MatchingABC
        A fitted MatchingABC object with no reference.

    knn_graph: Optional[List[csr_matrix]]
        List of knn-graphs, where knn-graph[i] is the knn-graph
        associated to matching.datasets[i]. Ignored if left empty.
    """
    # Retrieves information in case knn_graph is missing
    assert matchings.get((0, 1), None) is not None, "No matching provided."
    ndatasets = max(max(matchings.keys(), key=lambda k: max(k))) + 1
    sizes = [matchings[0, 1].shape[0]]
    sizes += [matchings[0, i].shape[1] for i in range(1, ndatasets)]
    # Loading these containers
    rows, cols, data = [], [], []
    # This mask allows to modify matching edges,
    # in order to increase their weight to select
    # them in priority compared to knn edges.
    rowsm, colsm, datam = [], [], []
    offset_i: int = 0
    for i in range(ndatasets):
        # Initial relations
        if knn_graphs is not None:
            knn_graph = knn_graphs[i].tocoo()
            rows += list(knn_graph.row + offset_i)
            cols += list(knn_graph.col + offset_i)
            data += list(knn_graph.data)
        offset_j: int = 0
        for j in range(ndatasets):
            if i == j:
                offset_j += sizes[i]
                continue
            T = matchings[i, j].tocoo()
            rows_k, cols_k, data_k = T.row, T.col, T.data
            rows_k += offset_i
            cols_k += offset_j
            rows += list(rows_k)
            cols += list(cols_k)
            data += list(data_k)
            rowsm += list(rows_k)
            colsm += list(cols_k)
            datam += [10000] * rows_k.shape[0]
            offset_j += sizes[j]
        offset_i += sizes[i]

    # We select only top weighted edges, and in priority the ones
    # from matching matrices by scaling them by a large factor
    N = sum(sizes)
    T = csr_matrix((data, (rows, cols)), shape=(N, N))
    T_mask = csr_matrix((data, (rows, cols)), shape=(N, N))
    ind, val = sort_sparse_matrix(T.multiply(T_mask), fill_empty=True, reverse=True)
    T = sparse_from_arrays(ind[:, :target_edges], val[:, :target_edges])
    T_mask.data = 1.0 / T_mask.data
    T = T.multiply(T_mask)
    T.data = np.clip(T.data, 0.0, 1.0)
    if symmetrize:
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
    distances = distances.copy()

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


def cluster_anndatas(
    datasets: List[AnnData],
    use_rep: Optional[str] = None,
    cluster_key: str = "cluster",
    n_neighbors: int = 15,
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
        adm.isset_value(adata, key=use_rep, field="obsm") for adata in datasets
    ), f"{use_rep} missing in .obsm of some AnnDatas."
    X = np.concatenate(
        [adata.obsm[use_rep] for adata in datasets],
        axis=0,
    )
    adj_matrix = nearest_neighbors_custom(X, mode="edges", n_neighbors=n_neighbors)
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


def node_geodesic_distances(adj: csr_matrix, directed: bool = True) -> np.ndarray:
    """
    Computes all distances between vertices of a graph
    represented as matrix $adj, where edges are weighted
    with length between vertices.

    TODO: additional parameters?
    """
    return dijkstra(adj, directed=directed)


def prune_edges_supervised(
    matchings: Dict[Tuple[int, int], csr_matrix],
    labels: List[np.ndarray],
) -> Dict[Tuple[int, int], csr_matrix]:
    """
    Selects only matching edges connected samples from the same class.
    """
    results = {}
    for i, j in matchings:
        T = matchings[i, j].tocoo()
        for k in range(T.data.shape[0]):
            row, col = T.row[k], T.col[k]
            if labels[i][row] != labels[j][col]:
                T.data[k] = 0.0
        T.eliminate_zeros()
        results[i, j] = T.tocsr()
    return results


def count_total_matching_edges(matchings: Dict[Tuple[int, int], csr_matrix]) -> int:
    nedges = 0
    for key in matchings:
        nedges += matchings[key].count_nonzero()
    return nedges


@njit()
def csr_get_row(
    indices: np.ndarray, K_indices: int, indptr: np.ndarray, K_indptrs: int, i: int
) -> np.ndarray:
    sj = indptr[i]
    if i == K_indptrs - 1:
        Sj = K_indices
    else:
        Sj = indptr[i + 1]
    return indices[sj:Sj]


@njit
def prune_non_transitive_njit(
    matching_indices: np.ndarray,
    matching_indices_K: np.ndarray,
    matching_indptrs: np.ndarray,
    matching_indptrs_K: np.ndarray,
    batch_i: int,
    batch_j: int,
    Tij_row: np.ndarray,
    Tij_col: np.ndarray,
    n_batches: int,
    min_patterns: int,
) -> np.ndarray:
    new_edges = np.zeros(Tij_row.shape)
    for idx in range(Tij_row.shape[0]):
        sample_i, sample_j = Tij_row[idx], Tij_col[idx]
        for batch_k in range(n_batches):
            if batch_k in (batch_i, batch_j):
                continue
            matches_j = csr_get_row(
                matching_indices[batch_j, batch_k],
                matching_indices_K[batch_j, batch_k],
                matching_indptrs[batch_j, batch_k],
                matching_indptrs_K[batch_j, batch_k],
                sample_j,
            )
            for sample_k in matches_j:
                matches_k = csr_get_row(
                    matching_indices[batch_k, batch_i],
                    matching_indices_K[batch_k, batch_i],
                    matching_indptrs[batch_k, batch_i],
                    matching_indptrs_K[batch_k, batch_i],
                    sample_k,
                )
                if sample_i in matches_k:
                    new_edges[idx] += 1
                    break
    return new_edges > min_patterns


def prune_edges_unsupervised(
    matchings: Dict[Tuple[int, int], csr_matrix],
    n_batches: int,
    min_patterns: int,
) -> Dict[Tuple[int, int], csr_matrix]:
    """
    Blabla
    """
    max_indices = np.max([matchings[key].indices.shape[0] for key in matchings])
    max_indptrs = np.max([matchings[key].indptr.shape[0] for key in matchings])

    # matching_indices[i, j] = Tij.indices, filled with -1 on the right
    # matching_indices_K[i, j] = # of non-(-1) values
    matching_indices = (
        np.zeros(shape=(n_batches, n_batches, max_indices), dtype=int) - 1
    )
    matching_indices_K = np.zeros(shape=(n_batches, n_batches), dtype=int)

    # matching_indptrs[i, j] = Tij.indptrs, filled with -1 on the right
    # matching_indptrs_K[i, j] = # of non-(-1) values
    matching_indptrs = (
        np.zeros(shape=(n_batches, n_batches, max_indptrs), dtype=int) - 1
    )
    matching_indptrs_K = np.zeros(shape=(n_batches, n_batches), dtype=int)

    for i in range(n_batches):
        for j in range(n_batches):
            if i == j:
                continue
            else:
                Tij = matchings[i, j]

                Kptr = Tij.indptr.shape[0]
                matching_indptrs_K[i, j] = Kptr
                matching_indptrs[i, j][0:Kptr] = Tij.indptr

                Kind = Tij.indices.shape[0]
                matching_indices_K[i, j] = Kind
                matching_indices[i, j][0:Kind] = Tij.indices

    result_matchings = {}
    for i, j in matchings:
        Tij_coo = matchings[i, j].tocoo()
        new_edges = prune_non_transitive_njit(
            matching_indices=matching_indices,
            matching_indices_K=matching_indices_K,
            matching_indptrs=matching_indptrs,
            matching_indptrs_K=matching_indptrs_K,
            batch_i=i,
            batch_j=j,
            Tij_col=Tij_coo.col,
            Tij_row=Tij_coo.row,
            n_batches=n_batches,
            min_patterns=min_patterns,
        )
        result_matchings[i, j] = Tij_coo.tocsr()
        result_matchings[i, j].data = new_edges
        result_matchings[i, j].eliminate_zeros()

    return result_matchings


@njit
def smooth_correction_vectors(
    X: np.ndarray,
    v: np.ndarray,
    nn_indices: np.ndarray,
    nn_distances: np.ndarray,
    n_neighbors: int = 5,
) -> np.ndarray:
    """
    Applies smoothing to a vector space.
    """
    new_correction = np.zeros(v.shape)
    for i, vi in enumerate(v):
        local_vs = v[nn_indices[i, :n_neighbors]]
        for j, v_nb_j in enumerate(local_vs):
            new_correction[i] += v_nb_j
    return new_correction / n_neighbors
