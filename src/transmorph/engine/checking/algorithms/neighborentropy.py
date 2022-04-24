#!/usr/bin/env python3

import numpy as np

from typing import List, Optional

from ..checking import Checking
from ...traits.usesneighbors import UsesNeighbors
from ...traits.isprofilable import profile_method
from ....stats.entropy import label_entropy
from ....utils.graph import nearest_neighbors


class NeighborEntropy(Checking, UsesNeighbors):
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

    def __init__(self, threshold: float = 0.5):
        Checking.__init__(self, str_identifier="NEIGHBOR_ENTROPY")
        UsesNeighbors.__init__(self)
        self.score: Optional[float] = None
        self.threshold = threshold

    def check_input(self, datasets: List[np.ndarray]) -> None:
        """
        Checks correct common dimensionality.
        """
        assert len(datasets) > 0
        dimensionality = datasets[0].shape[1]
        assert all(X.shape[1] == dimensionality for X in datasets)

    @profile_method
    def check(self, datasets: List[np.ndarray]) -> bool:
        """
        Computes label entropy, and return true if it is above threshold.
        """
        from transmorph import settings

        n_neighbors = settings.n_neighbors
        # Fraction of neighborhood conserved
        ndatasets = len(datasets)
        inner_nn_before = [
            self.get_neighbors_graph(i, mode="edges") for i in range(ndatasets)
        ]
        inner_nn_after = [nearest_neighbors(X, mode="edges") for X in datasets]
        nn_conserved = np.concatenate(  # For each point, ratio of kept neighbors
            [
                np.array(nnb.multiply(nna).sum(axis=1)) / n_neighbors
                for nna, nnb in zip(inner_nn_after, inner_nn_before)
            ],
            axis=0,
        )[:, 0]

        # Neighborhood entropy
        X_all = np.concatenate(datasets, axis=0)
        N = X_all.shape[0]
        T_all = nearest_neighbors(
            X_all,
            n_neighbors=n_neighbors,
            symmetrize=False,
        )
        belongs = np.zeros(N)
        offset = 0
        for i, X in enumerate(datasets, 1):
            n_obs = X.shape[0]
            belongs[offset : offset + n_obs] = i
            offset += n_obs
        T_all = T_all @ np.diag(belongs)
        # T_all: knn matrix with dataset ids as column values

        # origins: n*k matrix where o[i,j] if the dataset of origin
        # of the j-th neighbor of xi
        origins = np.zeros((N, n_neighbors))
        for i in range(N):
            oi = T_all[i]
            oi = oi[oi != 0]
            origins[i] = oi

        labels = list(set(origins.flatten()))
        entropies = label_entropy(origins, np.array(labels).astype(int))

        # Combining (n,) @ (n,)
        entropy = (nn_conserved @ entropies) / N
        self.score = entropy
        return self.score >= self.threshold
