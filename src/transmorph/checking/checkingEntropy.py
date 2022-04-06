#!/usr/bin/env python3

import numpy as np

from anndata import AnnData
from math import log
from numba import njit
from typing import List


from .checkingABC import CheckingABC
from ..utils.anndata_interface import get_matrix
from ..utils.graph import nearest_neighbors


@njit(fastmath=True)
def entropy(dataset_counts: np.ndarray, labels: np.ndarray):
    """Computes entropy of a discrete label distribution."""
    (npoints, k), nlabels = dataset_counts.shape, len(labels)
    entropies = np.zeros(npoints)
    if nlabels <= 1:
        return entropies
    probs = np.zeros((npoints, nlabels))
    for i in range(npoints):
        for lb in range(nlabels):
            probs[i, lb] = np.sum(dataset_counts[i] == lb + 1)
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


class CheckingEntropy(CheckingABC):
    """
    Uses a neighborhood criterion to provide a numerical estimation
    of neighborhood sanity after integration. It is computed as follows,

    CE(X) = 1/|X| * sum_{x \\in X} stability(x)*diversity(x),

    where stability(x) describes the neighborhood preservation of x
            before and after integration, and
          diversity(x) describes the heterogeneity of x neighbors
            after integration in terms of dataset of origin.

    stability(x) = |neighbors of x before and after|/|neighbors of x|
    diversity(x) = H(datasets in x neighborhood) where H is Shannon entropy

    Parameters
    ----------
    n_neighbors: int, default = 20
        Number of nearest neighbors to take into account when computing
        label entropy.
    """

    def __init__(
        self, threshold: float = 0.5, n_neighbors: int = 20, verbose: bool = False
    ):
        super().__init__(threshold=threshold, accept_if_lower=False, verbose=verbose)
        self.n_neighbors = n_neighbors

    def evaluate_metric(self, datasets: List[AnnData], X_kw: str = "") -> float:
        Xs_before = [get_matrix(adata, "") for adata in datasets]
        Xs_after = [get_matrix(adata, X_kw) for adata in datasets]

        # Fraction of neighborhood conserved
        inner_nn_before = [
            nearest_neighbors(
                X,
                n_neighbors=self.n_neighbors,
                symmetrize=False,
            )
            for X in Xs_before
        ]
        inner_nn_after = [
            nearest_neighbors(
                X,
                n_neighbors=self.n_neighbors,
                symmetrize=False,
            )
            for X in Xs_after
        ]
        nn_conserved = np.concatenate(  # For each point, ratio of kept neighbors
            [
                np.array(nnb.multiply(nna).sum(axis=1)) / self.n_neighbors
                for nna, nnb in zip(inner_nn_after, inner_nn_before)
            ],
            axis=0,
        )[:, 0]

        # Neighborhood entropy
        X_all = np.concatenate(Xs_after, axis=0)
        N = X_all.shape[0]
        T_all = nearest_neighbors(
            X_all,
            n_neighbors=self.n_neighbors,
            symmetrize=False,
        )
        belongs = np.zeros(N)
        offset = 0
        for i, adata in enumerate(datasets, 1):
            n_obs = adata.n_obs
            belongs[offset : offset + n_obs] = i
            offset += n_obs
        T_all = T_all @ np.diag(belongs)
        # T_all: knn matrix with dataset ids as column values

        # origins: n*k matrix where o[i,j] if the dataset of origin
        # of the j-th neighbor of xi
        origins = np.zeros((N, self.n_neighbors))
        for i in range(N):
            oi = T_all[i]
            oi = oi[oi != 0]
            origins[i] = oi

        labels = list(set(origins.flatten()))
        entropies = entropy(origins, np.array(labels).astype(int))

        # Combining
        return (nn_conserved @ entropies) / N
