#!/usr/bin/env python3

import numba
import numpy as np

from ..utils import sort_sparse_matrix
from ..utils.graph import nearest_neighbors


# Adapted from harmonypy:
# https://github.com/slowkow/harmonypy/blob/master/harmonypy/lisi.py
def compute_lisi(
    X: np.ndarray,
    labels: np.ndarray,
    perplexity: float = 15.0,
) -> np.ndarray:
    """
    LISI statistic measures how heterogeneous a sample neighborhood
    is for a certain label. Is is notably used in the Harmony
    integration pipeline to measure how well integrated datasets
    are.

    Parameters
    ----------
    X: np.ndarray
        (N, d) Concatenated data matrices in the embedding space

    labels: np.ndarray
        (N,) Labels of the observations

    perplexity: float, default = 30.0
        Neighborhood size.
    """

    # n_neighbors >= 3*perplexity
    connectivity = nearest_neighbors(
        X,
        n_neighbors=int(perplexity * 3),
        include_self_loops=False,
    )
    indices, _ = sort_sparse_matrix(connectivity)
    label_per_nb = labels[indices]

    all_labels = np.unique(labels)
    simpson = compute_simpson(label_per_nb, all_labels)

    return 1.0 / simpson


@numba.njit(fastmath=True)
def compute_simpson(label_per_nb: np.ndarray, labels_set: np.ndarray) -> np.ndarray:
    """
    Helper function for compute_lisi, returns simpson index for
    each sample, for one label
    """
    nlabels = labels_set.shape[0]
    nsamples, n_neighbors = label_per_nb.shape
    label_frequencies = np.zeros((nsamples, nlabels), dtype=np.float32)

    for i in range(nsamples):
        labels_i = label_per_nb[i]
        for lj, l in enumerate(labels_set):
            label_frequencies[i, lj] = (labels_i == l).sum()

    label_frequencies /= n_neighbors
    label_frequencies *= label_frequencies
    return np.sum(label_frequencies, axis=1)
