#!/usr/bin/env python3

import numpy as np

from numba import njit

from ..utils import sort_sparse_matrix
from ..utils.graph import nearest_neighbors


# Adapted from harmonypy:
# https://github.com/slowkow/harmonypy/blob/master/harmonypy/lisi.py
def compute_lisi(
    X: np.ndarray,
    labels: np.ndarray,
    perplexity: float = 30.0,
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
    indices, distances = sort_sparse_matrix(connectivity)

    # Compute Simpson index
    n_categories = len(np.unique(labels))
    simpson = compute_simpson(
        distances.T,
        indices.T,
        labels,
        n_categories,
        perplexity,
    )
    return 1.0 / simpson


@njit(fastmath=True)
def compute_simpson(
    distances: np.ndarray,
    indices: np.ndarray,
    labels: np.ndarray,
    perplexity: float,
    tol: float = 1e-5,
):
    """
    Helper function for compute_lisi, returns simpson index for
    each sample, for one label
    """
    n = distances.shape[1]
    simpson = np.zeros(n)
    logU = np.log(perplexity)
    # Loop through each cell.
    for i in range(n):
        beta = 1
        betamin = -np.inf
        betamax = np.inf
        # Compute Hdiff
        P = np.exp(-distances[:, i] * beta)
        P_sum = np.sum(P)
        if P_sum == 0:
            H = 0
            P = np.zeros(distances.shape[0])
        else:
            H = np.log(P_sum) + beta * np.sum(distances[:, i] * P) / P_sum
            P = P / P_sum
        Hdiff = H - logU
        n_tries = 50
        for _ in range(n_tries):
            # Stop when we reach the tolerance
            if abs(Hdiff) < tol:
                break
            # Update beta
            if Hdiff > 0:
                betamin = beta
                if not np.isfinite(betamax):
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if not np.isfinite(betamin):
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2
            # Compute Hdiff
            P = np.exp(-distances[:, i] * beta)
            P_sum = np.sum(P)
            if P_sum == 0:
                H = 0
                P = np.zeros(distances.shape[0])
            else:
                H = np.log(P_sum) + beta * np.sum(distances[:, i] * P) / P_sum
                P = P / P_sum
            Hdiff = H - logU
        # distancesefault value
        if H == 0:
            simpson[i] = -1
        # Simpson's index
        for label_category in np.unique(labels):
            ix = indices[:, i]
            q = labels[ix] == label_category
            if np.any(q):
                P_sum = np.sum(P[q])
                simpson[i] += P_sum * P_sum
    return simpson
