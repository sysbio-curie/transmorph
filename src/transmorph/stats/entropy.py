#!/usr/bin/env python3

from math import log
import numba
import numpy as np


@numba.njit(fastmath=True)
def label_entropy(sample_labels: np.ndarray, labels: np.ndarray) -> np.ndarray:
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
