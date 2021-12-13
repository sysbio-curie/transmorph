#!/usr/bin/env python3

from typing import Union
import numpy as np

from ot.lp import emd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from ..utils import sparse_cdist
from ..utils import nearest_neighbors


def matching_divergence(
    X1: np.ndarray,
    X2: np.ndarray,
    matching: csr_matrix,
    metric: str = "sqeuclidean",
    metric_kwargs: dict = {},
    per_point: bool = False,
    accelerated: bool = False,
) -> Union[float, np.ndarray]:
    """
    Computes the distorsion of an embedding between two datasets
    with known ground truth matching between samples.
    Given a boolean matching where M[i, j] = 1 if xi <-> yj,
    matching_divergence(X, Y, M) =

        1 / sum_ij (M[i, j]) * sum_ij M[i, j]*d(xi, yj)

    This is a basic measure of quadratic divergence (see MDE for instance).

    Parameters
    ----------
    X1: np.ndarray (n, d)
        First dataset embedded in a d-dimensional vector space

    X2: np.ndarray (m, d)
        Second dataset embedded in a d-dimensional vector space

    matching: scipy.sparse.csr_matrix (n, m)
        Boolean matching matrix between X1 and X2.

    metric: str, default = "sqeuclidean"
        Metric string to use for the distorsion.

    metric_kwargs: dict, default = {}
        Additional metric parameters

    accelerated: bool, default = False
        Use a sparse implementation instead of scipy.cdist for much higher
        speed. If so, metric can only be those supported by sparse_cdist
        from Transmorph.utils.geometry.
    """
    if accelerated:
        matched_distances = sparse_cdist(X1, X2=X2, T=matching, metric=metric)
    else:
        matched_distances = matching.multiply(
            cdist(X1, X2, metric=metric, **metric_kwargs)
        )
    matched_distances /= matching.count_nonzero()
    if per_point:
        return np.asarray(matched_distances.sum(axis=1)).reshape(
            -1,
        )
    return matched_distances.sum()


def neighborhood_preservation(
    X_before,
    X_after,
    n_neighbors: int = 10,
    metric: str = "sqeuclidean",
    metric_kwargs: dict = {},
    per_point: bool = False,
) -> Union[float, np.ndarray]:
    """
    Neighborhood preservation counts, for each point, the fraction of its
    initial nearest neighbors that stay nearest neighbors after integration.
    Therefore, it measures how well the initial dataset topology is preserved.

    Parameters
    ----------
    X_before: np.ndarray
        Dataset embedding before integration

    X_after: np.ndarray
        Dataset embedding after integration

    n_neighbors: int, default = 10
        Number of neighbors to take into account, the lower the more constraints

    metric: str, default = "sqeuclidean"
        Metric to use during kNN-graph computation

    metric_kwargs: dict, default = {}
        Dictionary containing additional metric parameters

    per_point: bool, default = False
        Return preservation per point intstead of global
    """
    assert (
        X_before.shape[0] == X_after.shape[0]
    ), "Number of samples must match between representations."
    if not per_point:
        return (
            neighborhood_preservation(
                X_before,
                X_after,
                n_neighbors=n_neighbors,
                metric=metric,
                per_point=True,
            ).sum()
            / X_before.shape[0]
        )
    nn_before = nearest_neighbors(
        X_before, n_neighbors=n_neighbors, metric=metric, metric_kwargs=metric_kwargs
    )
    nn_after = nearest_neighbors(X_after, n_neighbors=n_neighbors, metric=metric)
    return np.asarray(
        ((nn_before + nn_after) == 2.0).sum(axis=1) / n_neighbors
    ).reshape(
        -1,
    )


def earth_movers_distance(
    X1: np.ndarray,
    X2: np.ndarray,
    C: np.ndarray = None,
    metric: str = "sqeuclidean",
    metric_kwargs: dict = {},
    per_point: bool = False,
    max_iter: int = 1000000,
):
    """
    Earth movers distance (EMD) between X1 and X2 measures how well both
    embeddings overlap each other. We use a POT backend here to compute
    this measure. The higher EMD is, the more expansive it is to make
    both datasets overlap (and then, the poorer the overlap is).

    Parameters
    ----------
    X1: np.ndarray
        Source dataset after integration

    X2: np.ndarray
        Reference dataset

    C: np.ndarray, default = None
        Cost matrix between X1 and X2, if None then C is set to be
        the distance matrix between X1 and X2.

    metric: str = "squeuclidean"
        Metric to use if a distance matrix needs to be computed. It
        is passed as an argument to scipy.spatial.distance.cdist.

    metric_kwargs: dict, default = {}
        Additional metric parameters for cdist.

    per_point: bool, default = False
        If true, EMD per point is returned. Otherwise, the full cost
        is returned.

    max_iter: int, default = 1000000
        Number of iterations maximum for ot.lp.emd. Increase this
        number in case of convergence issues.
    """
    assert (
        C is not None or X1.shape[1] == X2.shape[1]
    ), "Explicit cost matrix needed if datasets are not in the same space."
    if C is None:
        C = cdist(X1, X2, metric=metric, **metric_kwargs)
    C /= C.max()
    w1, w2 = (np.ones(X1.shape[0]) / X1.shape[0], np.ones(X2.shape[0]) / X2.shape[0])
    T = emd(w1, w2, C, numItermax=max_iter)
    if per_point:
        return (T * C).sum(axis=1) * X1.shape[0]
    return (T * C).sum() * (X1.shape[0] * X1.shape[1])
