#!/usr/bin/env python3

import numpy as np

from scipy.sparse import csr_matrix

from ..utils.graph import nearest_neighbors


def neighborhood_preservation(
    X: np.ndarray,
    initial_neighbors: csr_matrix,
) -> float:
    """
    Neighborhood preservation counts, for each point, the fraction of its
    initial nearest neighbors that stay nearest neighbors after integration.
    Therefore, it measures how well the initial dataset topology is preserved.
    Returns the average over all points of this score.

    Parameters
    ----------
    X: np.ndarray
        Dataset embedding after integration

    initial_neighbors: csr_matrix
        Initial boolean k-NN matrix. k must be constant over each row.

    metric: str, default = "sqeuclidean"
        Metric to use during kNN-graph computation

    metric_kwargs: dict, default = {}
        Dictionary containing additional metric parameters
    """
    assert (
        X.shape[0] == initial_neighbors.shape[0] == initial_neighbors.shape[1]
    ), "Number of samples must match between representations."
    n_neighbors_all = np.array(initial_neighbors.sum(axis=1))[:, 0]
    n_neighbors = n_neighbors_all[0]
    assert all(n_neighbors_all == n_neighbors), "Non-constant number of neighbors."
    final_neighbors = nearest_neighbors(X, n_neighbors=int(n_neighbors), mode="edges")
    correct_nbs = initial_neighbors + final_neighbors
    return (correct_nbs == 2.0).sum() / initial_neighbors.count_nonzero()
