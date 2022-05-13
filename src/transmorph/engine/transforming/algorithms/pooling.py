#!/usr/bin/env python3

import numpy as np

from typing import List

from ..transformation import Transformation
from ...traits.usesneighbors import UsesNeighbors
from ....utils.matrix import pooling, extract_chunks, sort_sparse_matrix, perturbate
from ....utils.graph import nearest_neighbors


class Pooling(Transformation, UsesNeighbors):
    """
    Replaces each sample by the average of its neighbors. This
    is a useful to smooth data manifolds, and reduce the impact
    of outliers or artifacts. It can also be used after a merging
    to smooth the result, by toggling per_dataset parameter.

    Parameters
    ----------
    n_neighbors: int, default = 5
        Number of neighhbors to use for the pooling. The higher,
        the smoother data is.

    jitter_std: float, default = 0.01
        If > 0, applies a small perturbation at the end of pooling of
        standard deviation $jitter_std to unstick samples.

    per_dataset: bool, default = True
        Performs the pooling for each dataset independently. If false,
        all datasets are concatenated (they are then expected to be
        embedded in the same features space).
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        transformation_rate: float = 1.0,
        jitter_std: float = 0.01,
        per_dataset: bool = True,
    ):
        Transformation.__init__(self, "POOLING", True, transformation_rate)
        UsesNeighbors.__init__(self)
        self.n_neighbors = n_neighbors
        self.jitter_std = jitter_std
        self.per_dataset = per_dataset

    def transform(self, datasets: List[np.ndarray]) -> List[np.ndarray]:
        """
        Applies pooling, potentially partial.
        """
        if self.per_dataset:
            result = []
            for i, X in enumerate(datasets):
                indices, _ = self.get_neighbors_graph(
                    i,
                    "edges",
                    n_neighbors=self.n_neighbors,
                    return_format="arrays",
                )
                X_pooled = pooling(X, indices)
                if self.transformation_rate <= 1.0:
                    X_pooled *= self.transformation_rate
                    X_pooled += X * (1.0 - self.transformation_rate)
                result.append(X_pooled)
        else:
            X_all = np.concatenate(datasets, axis=0)
            nn_matrix = nearest_neighbors(
                X_all,
                n_neighbors=self.n_neighbors,
                mode="edges",
            )
            indices, _ = sort_sparse_matrix(nn_matrix)
            X_pooled = pooling(X_all, indices)
            if self.transformation_rate <= 1.0:
                X_pooled *= self.transformation_rate
                X_pooled += X_all * (1.0 - self.transformation_rate)
            result = extract_chunks(X_pooled, [X.shape[0] for X in datasets])

        if self.jitter_std > 0:
            for X in result:
                X = perturbate(X, std=self.jitter_std)

        return result
