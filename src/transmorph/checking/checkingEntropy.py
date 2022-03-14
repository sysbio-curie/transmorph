#!/usr/bin/env python3

import numpy as np

from anndata import AnnData
from math import log, e
from typing import List


from .checkingABC import CheckingABC
from ..utils.anndata_interface import get_matrix
from ..utils.graph import nearest_neighbors


# Shamelessly borrowed from
# https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
def entropy(labels):
    """Computes entropy of label distribution."""

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    _, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.0

    # Compute entropy
    for i in probs:
        ent -= i * log(i, e)

    return ent


class CheckingEntropy(CheckingABC):
    """
    TODO
    """

    def __init__(
        self, threshold: float = 0.5, n_neighbors: int = 20, verbose: bool = False
    ):
        super().__init__(threshold=threshold, accept_if_lower=False, verbose=verbose)
        self.n_neighbors = 20

    def evaluate_metric(self, datasets: List[AnnData], X_kw: str = "") -> float:
        Xs_before = [get_matrix(adata, "") for adata in datasets]
        Xs_after = [get_matrix(adata, X_kw) for adata in datasets]

        # Fraction of neighborhood conserved
        use_nndescent = any(X.shape[0] > 2048 for X in Xs_before)  # Large datasets?
        inner_nn_before = [
            nearest_neighbors(
                X,
                n_neighbors=self.n_neighbors,
                use_nndescent=use_nndescent,
                symmetrize=False,
            )
            for X in Xs_before
        ]
        inner_nn_after = [
            nearest_neighbors(
                X,
                n_neighbors=self.n_neighbors,
                use_nndescent=use_nndescent,
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
        use_nndescent = N > 2048
        T_all = nearest_neighbors(
            X_all,
            n_neighbors=self.n_neighbors,
            use_nndescent=use_nndescent,
            symmetrize=False,
        )
        belongs = np.zeros(N)
        offset = 0
        for i, adata in enumerate(datasets, 1):
            n_obs = adata.n_obs
            belongs[offset : offset + n_obs] = i
            offset += n_obs
        T_all = T_all @ np.diag(belongs)

        # TODO: optimize this part
        entropies = np.zeros(N)
        for i in range(N):
            origins = T_all[i]
            origins = origins[origins != 0]
            entropies[i] = entropy(origins)

        # Combining
        return (nn_conserved @ entropies) / N
