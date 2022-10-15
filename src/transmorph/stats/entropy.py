#!/usr/bin/env python3

import numba
import numpy as np

from math import log
from scipy.sparse import diags

from ..utils.graph import nearest_neighbors_custom
from ..utils.matrix import sort_sparse_matrix


@numba.njit(fastmath=True)
def _label_entropy_njit(sample_labels: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Computes entropy of a discrete label distribution. Numba-accelerated.

    Parameters
    ----------
    sample_labels: np.ndarray
        Matrix of shape (n_samples, k) where M[i, j] is the label of
        point $j in sample $i.

    labels: np.ndarray
        Vector of shape (n_labels,) containing all possible labels

    Returns
    -------
    A np.ndarray of shape (n_samples,) that contains for each sample its
    label entropy.
    """
    (npoints, k), nlabels = sample_labels.shape, labels.shape[0]
    entropies = np.zeros(npoints)
    if nlabels <= 1:  # No label diversity
        return entropies
    # M[i, l] is the frequency of label $l in sample $i
    probs = np.zeros((npoints, nlabels))
    for i, lb in enumerate(labels):
        probs[:, i] = np.sum(sample_labels == lb, axis=1)
    probs /= k
    for i in range(npoints):
        n_classes = np.count_nonzero(probs[i])
        if n_classes <= 1:
            continue
        for p in probs[i]:
            if p == 0.0:
                continue
            entropies[i] -= p * log(p)
    return entropies


def label_entropy(X: np.ndarray, labels: np.ndarray, n_neighbors: int) -> float:
    """
    Computes label entropy from a nearest neighbors matrix.
    """
    N = X.shape[0]

    # We build a kNN matrix where values are initial labels.
    T_all = nearest_neighbors_custom(X, mode="edges", n_neighbors=n_neighbors)
    T_all = T_all @ diags(labels)

    _, origins = sort_sparse_matrix(T_all)
    labels_set = list(set(origins.flatten()))
    entropies = _label_entropy_njit(origins, np.array(labels_set).astype(int))

    return entropies.sum() / N
