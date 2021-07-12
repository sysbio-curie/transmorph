#!/usr/bin/env python3

import numpy as np

from numba import njit
from scipy.sparse import coo_matrix, csr_matrix


def col_normalize(x: np.ndarray):
    x_std = x.std(axis=0)
    x_std[x_std == 0] = 1
    return x / x_std


def symmetrize(L):
    coox, cooy, coodata = symmetrize_csr_to_coo(
        L.indptr,
        L.indices,
        L.data
    )
    return coo_matrix(
        (coodata, (coox, cooy)),
        shape=L.shape
    ).tocsr()
    

@njit
def symmetrize_csr_to_coo(ptrs, inds, data):
    # Necessary for numba compatibility (does not support sparse)
    
    mapping = {}
    # preprocessing nnz values
    for i in range(len(ptrs)):
        m = ptrs[i]
        M = ptrs[i+1] if i < len(ptrs) - 1 else len(inds)
        for j, v in zip(inds[m:M], data[m:M]):
            mapping[(i,j)] = v

    # Filling loop
    xs, ys, vs = [], [], []
    for i in range(len(ptrs)):
        m = ptrs[i]
        M = ptrs[i+1] if i < len(ptrs) - 1 else len(inds)
        for j, v in zip(inds[m:M], data[m:M]):
            v1 = mapping[(i,j)]
            v2 = mapping.get((i,j), 0)
            if v2 != 0:
                v1 = (v1 + v2) / 2
            xs.append(i)
            ys.append(j)
            vs.append(v)
            if i != j:
                xs.append(j)
                ys.append(i)
                vs.append(v)

    return xs, ys, vs


@njit
def vertex_cover(Lptr, Linds, hops=2):
    # Lptr, Linds: CSR matrix representation
    # of neighbors
    n = Lptr.shape[0] - 1
    anchors = np.zeros(n, dtype='int')
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
                M = (Lptr[nb+1] if nb + 1 < n else n)
                for nb2 in Linds[Lptr[nb]:M]:
                    if anchors[nb2]:
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
