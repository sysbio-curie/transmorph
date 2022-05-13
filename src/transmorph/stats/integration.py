#!/usr/bin/env python3

import numpy as np
from typing import Dict, Optional, Union

from ot.lp import emd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from ..utils.geometry import sparse_cdist

# TODO: update to better compare between datasets


def matching_divergence(
    X1: np.ndarray,
    X2: np.ndarray,
    matching: csr_matrix,
    metric: str = "sqeuclidean",
    metric_kwargs: Optional[Dict] = None,
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
    metric_kwargs = {} if metric_kwargs is None else metric_kwargs
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


def earth_movers_distance(
    X1: np.ndarray,
    X2: np.ndarray,
    C: Optional[np.ndarray] = None,
    metric: str = "sqeuclidean",
    metric_kwargs: Optional[Dict] = None,
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

    C: np.ndarray, optional
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
    metric_kwargs = {} if metric_kwargs is None else metric_kwargs
    if C is None:
        C = cdist(X1, X2, metric=metric, **metric_kwargs)
    C /= C.max()
    w1, w2 = (np.ones(X1.shape[0]) / X1.shape[0], np.ones(X2.shape[0]) / X2.shape[0])
    T = emd(w1, w2, C, numItermax=max_iter)
    if per_point:
        return (T * C).sum(axis=1) * X1.shape[0]
    return (T * C).sum() * (X1.shape[0] * X1.shape[1])
