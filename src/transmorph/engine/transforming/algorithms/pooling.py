#!/usr/bin/env python3

import numpy as np

from typing import List

from ..transformation import Transformation
from ...traits.usesneighbors import UsesNeighbors
from ....utils.matrix import pooling


class Pooling(Transformation, UsesNeighbors):
    """
    Replaces each sample by the average of its neighbors. This
    is a useful to smooth data manifolds, and reduce the impact
    of outliers or artifacts.

    Parameters
    ----------
    n_neighbors: int, default = 5
        Number of neighhbors to use for the pooling. The higher,
        the smoother data is.
    """

    def __init__(self, n_neighbors: int = 5):
        Transformation.__init__(self, True, "POOLING")
        UsesNeighbors.__init__(self)
        self.n_neighbors = n_neighbors

    def transform(self, datasets: List[np.ndarray]) -> List[np.ndarray]:
        result = []
        for i, X in datasets:
            indices, _ = self.get_neighbors_graph(
                i,
                "edges",
                n_neighbors=self.n_neighbors,
            )
            pooling(X, indices)
        return result
