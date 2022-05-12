#!/usr/bin/env python3

import numpy as np

from typing import List

from ..checking import Checking
from ...traits.isprofilable import profile_method
from ....stats.entropy import label_entropy


class NeighborEntropy(Checking):
    """
    Uses a neighborhood criterion to provide a numerical estimation
    of neighborhood sanity after integration. Intuitively, it attributes
    a score to each sample, that penalizes points whose neighbors in
    the embedding space come from the same dataset.

    For a set of datasets D = {X1, ..., XN} embedded in the same space
    using a projector f,

    NE(S) = (|X1| + ... + |XN|)^{-1}
            \\times \\sum_{Xi \\in S} \\sum_{x \\in xi}
            H_k(NN(f(x), f(X1) \\cup ... \\cup f(XN))),

    Where H_k(x, S) is the Shannon entropy of labels in the k-nearest
    neighbors of x (including x).

    Parameters
    ----------
    n_neighbors: int, default = 15
        Number of nearest neighbors to take into account when computing
        label entropy.

    threshold: float, default = 0.5
        check() is accepted if neighbor entropy is above this threshold.
    """

    def __init__(self, n_neighbors: int = 15, threshold: float = 0.5):
        Checking.__init__(self, str_identifier="NEIGHBOR_ENTROPY")
        self.n_neighbors = n_neighbors
        self.threshold = threshold

    def check_input(self, datasets: List[np.ndarray]) -> None:
        """
        Checks correct common dimensionality of the embeddings.
        """
        assert len(datasets) > 0
        dimensionality = datasets[0].shape[1]
        assert all(X.shape[1] == dimensionality for X in datasets)

    @profile_method
    def check(self, datasets: List[np.ndarray]) -> bool:
        """
        Computes label entropy, and returns true if it is above threshold.
        """
        # Concatenating datasets and labels
        X_all = np.concatenate(datasets, axis=0)
        N = X_all.shape[0]

        labels = np.zeros(N)
        offset = 0
        for i, X in enumerate(datasets, 1):
            n_obs = X.shape[0]
            labels[offset : offset + n_obs] = i
            offset += n_obs

        self.score = label_entropy(X_all, labels, self.n_neighbors)
        return self.score >= self.threshold
