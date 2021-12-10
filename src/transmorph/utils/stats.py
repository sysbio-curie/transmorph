#!/usr/bin/env python3

import numpy as np
from scipy.sparse import csr_matrix
from ..utils import sparse_cdist


def matching_divergence(
    X1: np.ndarray,
    X2: np.ndarray,
    matching: csr_matrix,
    metric: str = "sqeuclidean",
):
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
    """
    return (
        sparse_cdist(X1, X2=X2, T=matching, metric=metric).sum()
        / matching.count_nonzero()
    )
