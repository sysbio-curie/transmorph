#!/usr/bin/env python3

import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist
from scipy import sparse
import osqp

def normal_kernel_weights(
    x: np.ndarray, scale: float = 1, alpha_qp: float = 1.0
):

    assert scale > 0, "sigma must be positive."
    assert 0 < alpha_qp < 2, "alpha_qp must be in (0,2)"

    Dmatrix = cdist(x, x)

    assert Dmatrix.max() > 0, "All points are equal in x."
    Dmatrix /= Dmatrix.std()
    
    K = norm.pdf(-Dmatrix, loc=0, scale=scale)
    return _optimal_weights(K, alpha_qp)


def _optimal_weights(K: np.ndarray, alpha_qp: float = 1.0, eps=1e-9):
    """ Computes optimal weights given K pairwise kernel matrix. """

    # Cost matrix
    M = K.mean(axis=1)
    KM = K - M
    P = sparse.csc_matrix(2 * np.dot(KM.T, KM))  # Compensate the 1/2 in QP formulation
    n = len(KM)

    # Define problem data
    q = np.array([0] * n)
    A = np.identity(n)
    A = sparse.csc_matrix(np.vstack((A, np.array([1] * n))))
    l = np.array([0] * n + [1])
    u = np.array([1] * (n + 1))

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace and change alpha parameter
    prob.setup(P, q, A, l, u, alpha=alpha_qp, verbose=False)

    # Solve problem
    x = prob.solve().x
    x = np.clip(x, eps, 1)
    return x / x.sum()
