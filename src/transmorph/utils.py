#!/usr/bin/env python3

import numpy as np

from numba import njit
from numpy.random import RandomState
from pynndescent import NNDescent
from scipy.sparse import coo_matrix, csr_matrix
from umap.umap_ import fuzzy_simplicial_set


@njit
def _get_row_indices(ptr, ind, i):
    start = ptr[i]
    end = len(ind) if i == len(ptr) - 1 else ptr[i + 1]
    return ind[start:end]


@njit
def _get(ptr, ind, dat, i, j):
    # Emulates M[i,j] for sparse matrices in numba
    start = ptr[i]
    end = len(ind) if i == len(ptr) - 1 else ptr[i + 1]
    for k, jprime in enumerate(ind[start:end]):
        if jprime == j:
            return dat[start + k]
    return 0.0


# CSR matrices must be decomposed as np.ndarrays for
# numba compatibility :'-(
# @njit(fastmath=True) TODO
def _combine_graphs(
        # G1 (n, n)
        ptr1,
        ind1,
        dat1,
        # G2 (m, m)
        ptr2,
        ind2,
        dat2,
        # G12 (n0, m0)
        ptr12,
        ind12,
        dat12,
        # G21 (n0, m0)
        ptr21,
        ind21,
        dat21,
        # anchors
        x1_anchors,
        x1_mapping,
        # x1_strengths, TODO: add the w(a_i <-> a_ik)
        x2_anchors,
        x2_mapping,
        # x2_strengths, TODO: add the w(a_i <-> a_ik)
        n_neighbors 
):
    (n, m) = (ptr1.shape[0] - 1, ptr2.shape[0] - 1)
    (n0, m0) = int(x1_anchors.sum()), int(x2_anchors.sum())
    final_indices = np.zeros((n + m, n_neighbors), dtype=np.int64) - 1

    # Handy mappings
    anchor_to_index_1 = np.arange(n)[x1_anchors]
    anchor_to_index_2 = np.arange(m)[x2_anchors]
    index_to_anchor_1 = {i: k for k, i in enumerate(anchor_to_index_1)}
    index_to_anchor_2 = {i: k for k, i in enumerate(anchor_to_index_2)}
    anchor_mapping_x = {}
    for i, a_i in enumerate(x1_mapping):
        if a_i not in anchor_mapping_x:
            anchor_mapping_x[a_i] = []
        if i != a_i:
            anchor_mapping_x[a_i].append(i)
    anchor_mapping_y = {}
    for j, b_j in enumerate(x2_mapping):
        if b_j not in anchor_mapping_y:
            anchor_mapping_y[b_j] = []
        if j != b_j:
            anchor_mapping_y[b_j].append(j)
    
    # Gathering edges from both graphs
    for i in range(n):
        neighbors = _get_row_indices(ptr1, ind1, i).astype(np.int32) # Original matrix
        if x1_anchors[i]:
            idx_i = index_to_anchor_1[i]
            indices_nn21 = _get_row_indices(ptr21, ind21, idx_i)
            indices_nn12 = _get_row_indices(ptr12, ind12, idx_i)
            common_indices = list(set(indices_nn12) & set(indices_nn21))
            for k, idx_nb in enumerate(common_indices):
                if idx_nb < n0:
                    common_indices[k] = anchor_to_index_1[idx_nb]
                else:
                    common_indices[k] = anchor_to_index_2[idx_nb - n0]
            neighbors = np.array(
                list(set(neighbors) | set(common_indices)),
                dtype=np.int32
            )
        final_indices[i,:len(neighbors)] = neighbors
    for i in range(m):
        neighbors = _get_row_indices(ptr2, ind2, i) + n # Original matrix
        neighbors = neighbors.astype(np.int32)
        if x2_anchors[i]:
            idx_i = index_to_anchor_2[i]
            indices_nn21 = _get_row_indices(ptr21, ind21, n0 + idx_i)
            indices_nn12 = _get_row_indices(ptr12, ind12, n0 + idx_i)
            common_indices = list(set(indices_nn12) & set(indices_nn21))
            for k, idx_nb in enumerate(common_indices):
                if idx_nb < n0:
                    common_indices[k] = anchor_to_index_1[idx_nb]
                else:
                    common_indices[k] = anchor_to_index_2[idx_nb - n0]
            neighbors = np.array(
                list(set(neighbors) | set(common_indices)),
                dtype=np.int32
            )
        final_indices[n + i,:len(neighbors)] = neighbors
    
    # Building graph
    # (i,j) is an edge if:
    # - (i,j) \in G1, or
    # - (i,j) \in G2, or
    # - (i,j) \in G12 and G21, we weight it (wij)12*(wij)21
    rows, cols, data = [], [], []
    for i in range(n + m):
        for j in final_indices[i]:
            if i < j or j == -1:
                continue       
            val = 0.0
            if i < n and j < n:
                val = _get(ptr1, ind1, dat1, i, j)
                if x1_anchors[i] and x1_anchors[j]:
                    idx_i, idx_j = (
                        index_to_anchor_1[i],
                        index_to_anchor_1[j]
                    )
                    val12 = _get(ptr12, ind12, dat12, idx_i, idx_j)
                    val21 = _get(ptr21, ind21, dat21, idx_i, idx_j)
                    val = (
                        val + val12 + val21
                        - val*val12 - val*val21 - val12*val21
                        + val*val12*val21
                    )
            elif i >= n and j >= n:
                val = _get(ptr2, ind2, dat2, i - n, j - n)
                if x2_anchors[i - n] and x2_anchors[j - n]:
                    idx_i, idx_j = (
                        index_to_anchor_2[i - n],
                        index_to_anchor_2[j - n]
                    )
                    val12 = _get(ptr12, ind12, dat12, idx_i, idx_j)
                    val21 = _get(ptr21, ind21, dat21, idx_i, idx_j)
                    val = (
                        val + val12 + val21
                        - val*val12 - val*val21 - val12*val21
                        + val*val12*val21
                    )
            else: # i >= n, j < n
                if x1_anchors[j] and x2_anchors[i - n]:
                    idx_i, idx_j = (
                        n0 + index_to_anchor_2[i - n],
                        index_to_anchor_1[j]
                    )
                    val12 = _get(ptr12, ind12, dat12, idx_i, idx_j)
                    val21 = _get(ptr21, ind21, dat21, idx_i, idx_j)
                    val = val12 + val21 - val12 * val21
            if val > 0.0:
                rows.append(i)
                rows.append(j)
                cols.append(j)
                cols.append(i)
                data.append(val)
                data.append(val)
                if i < n or j >= n:
                    continue
                # If a_i <-> b_j then we link all a_i-dependent points
                # to b_j and vice-versa
                for ai_nb in anchor_mapping_x[j]:
                    rows.append(ai_nb)
                    rows.append(i)
                    cols.append(i)
                    cols.append(ai_nb)
                    data.append(val)
                    data.append(val)
                for bj_nb in anchor_mapping_y[i - n]:
                    rows.append(j)
                    rows.append(bj_nb + n)
                    cols.append(bj_nb + n)
                    cols.append(j)
                    data.append(val)
                    data.append(val)

    return rows, cols, data


def within_modality_stabilize(X, indices, n_neighbors=15):
    X_stabilized = X.copy()
    for i in range(X.shape[0]):
        X_stabilized[i] = np.sum(X[indices[i]], axis=0)/n_neighbors
    return X_stabilized


def combine_graphs(
        G1,
        G2,
        G12,
        G21,
        x_anchors,
        x_mapping,
        y_anchors,
        y_mapping,
        n_neighbors
):
    
    n_neighbors = max(
        (G1 > 0).sum(axis=0).max(),
        (G2 > 0).sum(axis=0).max(),
        (G12 > 0).sum(axis=0).max(),
        (G21 > 0).sum(axis=0).max(),
    )
    # Creating adjacency matrix
    rows, cols, data = _combine_graphs(
        G1.indptr,
        G1.indices,
        G1.data,
        G2.indptr,
        G2.indices,
        G2.data,   
        G12.indptr,
        G12.indices,
        G12.data,
        G21.indptr,
        G21.indices,
        G21.data,
        x_anchors,
        x_mapping,
        y_anchors,
        y_mapping,
        3*n_neighbors
    )

    N = G1.shape[0] + G2.shape[0]
    return coo_matrix(
        (data, (rows, cols)),
        shape=(N, N),
        dtype=np.float32
    ).tocsr()


def compute_umap_graph(
        X,
        metric: str = "euclidean",
        metric_kwargs: dict = {},
        random_seed: int = 42,
        n_neighbors: int = 15,
        min_iters: int = 5,
        min_trees: int = 64,
        max_candidates: int = 60,
        set_op_mix_ratio: float = 1.0,
        local_connectivity: float = 1.0,
        low_memory: bool = False,
        n_jobs: int = -1,
):
    # TODO: disconnection distance
    # Borrowed from UMAP's implementation
    # https://github.com/lmcinnes/umap
    n_trees = min(min_trees, 5 + int(round((X.shape[0]) ** 0.5 / 20.0)))
    n_iters = max(min_iters, int(round(np.log2(X.shape[0]))))
    knn_search_index = NNDescent(
        X,
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=RandomState(random_seed),
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=max_candidates,
        low_memory=low_memory,
        n_jobs=n_jobs,
        verbose=False,
    )
    knn_indices, knn_dists = knn_search_index.neighbor_graph
    
    (
        strengths,
        sigmas,
        rhos,
        dists
    ) = fuzzy_simplicial_set(
        X,
        n_neighbors,
        random_seed,
        metric,
        metric_kwds=metric_kwargs,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        angular=False,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
        apply_set_operations=True,
        verbose=False,
        return_dists=True,
    )
    return {
        "knn_indices": knn_indices,
        "knn_distances": knn_dists,
        "strengths": strengths,
        "sigmas": sigmas,
        "rhos": rhos,
        "fuzzy_distances": dists
    }


def col_normalize(x: np.ndarray):
    x_std = x.std(axis=0)
    x_std[x_std == 0] = 1
    return (x - x.mean(axis=0)) / x_std


def symmetrize_distance_matrix(indices, dists):
    n = indices.shape[0]
    rows, cols, data = fill_coo_matrix(indices, dists)
    A = coo_matrix(
        (data, (rows, cols)), shape=(n, n)
    ).tocsr() # A[i,j] = d(i, j) if j nn of i
    # Symmetrizing
    A_mask = A.copy()
    A_mask.data = np.ones((A_mask.data.shape[0],), dtype=np.bool8)
    A_mask = A_mask * A_mask.T # =1 if (i,j) and (j,i) \in A
    return A + A.T - .5 * (A * A.T * A_mask)


def symmetrize_strength_matrix(indices, strengths):
    n = indices.shape[0]
    rows, cols, data = fill_coo_matrix(indices, strengths)
    A = coo_matrix(
        (data, (rows, cols)), shape=(n, n)
    ).tocsr() # A[i,j] = d(i, j) if j nn of i
    # Symmetrizing
    return A + A.T - A * A.T


@njit
def fill_coo_matrix(indices, values):
    rows, cols, data = [], [], []
    for i, (ni, vi) in enumerate(zip(indices, values)):
        for nij, vij in zip(ni, vi):
            rows.append(i)
            cols.append(nij)
            data.append(vij)
    return rows, cols, data


@njit
def vertex_cover(Lptr, Linds, hops=1):
    # Lptr, Linds: CSR matrix representation
    # of neighbors
    n = Lptr.shape[0] - 1 # /!\ -1 because of CSR representation
    anchors = np.zeros(n, dtype='int') - 1
    for v in range(n):
        # If v is visited, nothing to do
        if anchors[v] != -1:
            continue
        # If v not visited, it is its own anchor
        anchors[v] = v
        # Mark its neighbors as visited
        neighbors = [(v, 0)]
        while len(neighbors):
            nb, d = neighbors.pop()
            anchors[nb] = v
            if d < hops:
                M = (Lptr[nb+1] if nb + 1 < n else n)
                for nb2 in Linds[Lptr[nb]:M]:
                    if anchors[nb2] != -1:
                        continue
                    anchors[nb2] = v
                    neighbors.append( (nb2, d+1) )
    anchors_set = np.zeros(n)
    for i in set(anchors):
        anchors_set[i] = 1
    return anchors_set, anchors # set, map


def weight_per_label(xs_labels, yt_labels):
    # Weighting points by label proportion, so that
    # - Label total weights is equal in each dataset
    # - Dataset-specific labels weight 0

    n, m = len(xs_labels), len(yt_labels)
    all_labels = list(set(xs_labels).union(set(yt_labels)))

    # Labels specific to xs/yt
    labels_specx = [ i for (i, li) in enumerate(all_labels)
                     if li not in yt_labels ]
    labels_specy = [ i for (i, li) in enumerate(all_labels)
                     if li not in xs_labels ]
    labels_common = [ i for (i, li) in enumerate(all_labels)
                      if i not in labels_specx
                      and i not in labels_specy ]

    # Fequency of each label
    xs_freqs = np.array([
        np.sum(xs_labels == li) / n for li in all_labels
    ])
    yt_freqs = np.array([
        np.sum(yt_labels == li) / m for li in all_labels
    ])

    # Only accounting for common labels
    norm_x, norm_y = (
        np.sum(xs_freqs[labels_common]),
        np.sum(yt_freqs[labels_common])
    )
    rel_freqs = np.zeros(len(all_labels))
    rel_freqs[labels_common] = (
        yt_freqs[labels_common] * norm_x / (xs_freqs[labels_common] * norm_y)
    )

    # Correcting weights with respect to label frequency
    wx, wy = np.ones(n) / n, np.ones(m) / m
    for fi, li in zip(rel_freqs, all_labels):
        wx[xs_labels == li] *= fi
    for i in labels_specx + labels_specy:
        wy[yt_labels == all_labels[i]] = 0

    return wx / wx.sum(), wy / wy.sum()


@njit
def fill_dmatrix(D_anchors, anchors, anchors_map, distances_map):
    # D_i,j ~ D_ri,rj + D_i,ri + D_j,rj
    n = len(anchors_map)
    D = np.zeros((n,n), dtype=np.float32)
    small_idx = [
        np.sum(anchors[:anchors_map[i]])
        for i in range(n)
    ]
    D = (
        D_anchors[small_idx, small_idx] 
        + distances_map 
        + distances_map[:,None]
    )
    return D - np.diag(np.diag(D))
