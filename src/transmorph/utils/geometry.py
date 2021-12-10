#!/usr/bin/env python3

from logging import warn
from numba.core.decorators import njit
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial.distance import cdist


def sparse_cdist(
    X1: np.ndarray,
    X2: np.ndarray = None,
    T: csr_matrix = None,
    metric: str = "sqeuclidean",
) -> csr_matrix:
    if X2 is None:
        X2 = X1
    if T is None:
        warn(
            "Using sparse_cdist without a matching is useless."
            "Use scipy's cdist instead."
        )
        return csr_matrix(cdist(X1, X2, metric=metric))
    T = T.tocoo()
    if metric == "sqeuclidean":
        dfunc = dfunc_sqeuclidean
    elif metric == "euclidean":
        dfunc = dfunc_euclidean
    else:
        raise NotImplementedError("Unrecognized metric.")
    data = sparse_cdist_njit(X1, X2, T.row, T.col, dfunc)
    return coo_matrix((data, (T.row, T.col)), shape=T.shape).tocsr()


@njit(fastmath=True)
def sparse_cdist_njit(X1, X2, row, col, dfunc):
    data = np.zeros(row.shape)
    for k, (i, j) in enumerate(zip(row, col)):
        data[k] = dfunc(X1[i], X2[j])
    return data


@njit
def dfunc_sqeuclidean(x, y):
    return (x - y) @ (x - y)


@njit
def dfunc_euclidean(x, y):
    return np.sqrt(dfunc_sqeuclidean(x, y))
