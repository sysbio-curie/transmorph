#!/usr/bin/env python3

import numpy as np

from scipy.sparse import csr_matrix
from typing import List, Literal

from ..checking import Checking
from ...traits.usesneighbors import UsesNeighbors
from ...traits.isprofilable import profile_method
from ....stats.neighbors import neighborhood_preservation


class NeighborConservation(Checking, UsesNeighbors):
    """
    Uses a neighborhood criterion to provide a numerical estimation
    of neighborhood sanity after integration. It will penalize the
    integrated datasets where neighborhood relations are not preserved
    after integration.

    For a dataset embedding X of n samples and a projector f in the
    integration space, we compute two k-nearest neighbors graphs

    Ni = NN(X, k)
    Nf = NN(f(X), k)

    Then, neighbor conservation is given by 1 - |Ni - Nf| / (k * n)

    Parameters
    ----------
    n_neighbors: int, default = 15
        Number of nearest neighbors to take into account when computing
        label entropy.

    threshold: float, default = 0.5
        check() is accepted if neighbor entropy is above this threshold.

    mode: Literal["min", "max", "mean"] = "mean"
        How to aggregate results across datasets.
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        threshold: float = 0.5,
        mode: Literal["min", "max", "mean"] = "mean",
    ):
        Checking.__init__(self, str_identifier="NEIGHBOR_ENTROPY")
        UsesNeighbors.__init__(self)
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        assert mode in ("min", "max", "mean"), f"Unknown mode {mode}."
        self.mode = mode

    def check_input(self, datasets: List[np.ndarray]) -> None:
        """
        Checks correct common dimensionality of the embeddings.
        """
        assert len(datasets) > 0

    @profile_method
    def check(self, datasets: List[np.ndarray]) -> bool:
        """
        Computes label entropy, and returns true if it is above threshold.
        """
        scores = []
        for i, X in enumerate(datasets):
            neighbors = UsesNeighbors.get_neighbors_graph(i, "edges", self.n_neighbors)
            assert isinstance(neighbors, csr_matrix)
            scores.append(neighborhood_preservation(X, neighbors))
        if self.mode == "mean":
            self.score = np.mean(scores)
        elif self.mode == "min":
            self.score = np.min(scores)
        elif self.mode == "max":
            self.score = np.max(scores)
        else:
            raise ValueError(f"Unknown mode {self.mode}.")
        return self.score >= self.threshold
