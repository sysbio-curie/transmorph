#!/usr/bin/env python3

import numpy as np
from awkde import GaussianKDE
from scipy.stats import multivariate_normal
import math
from scipy import sparse
import osqp


def _gaussian_kernel_matrix(x: np.ndarray, scale: float = 1, alpha: float = 0.5):
    """ Computes the (n,n) pairwise kernel matrix k_j(x_i) """
    kde = GaussianKDE(glob_bw="silverman", alpha=alpha, diag_cov=True)
    m, S = kde.fit(x)
    li = kde._inv_loc_bw

    n = len(x)
    K_np = np.zeros((n, n))
    for i in range(n):
        K_np[:, i] = multivariate_normal.pdf(x, mean=x[i], cov=li[i] * S * scale)
    return K_np


def normal_kernel_weights(
    x: np.ndarray, scale: float = 1, alpha_kde: float = 0.5, alpha_qp: float = 1.0
):

    assert scale > 0, "sigma must be positive."
    assert 0 < alpha_kde < 1, "alpha_kde must be in (0,1)"
    assert 0 < alpha_qp < 2, "alpha_qp must be in (0,2)"

    K = _gaussian_kernel_matrix(x, scale, alpha_kde)
    return _optimal_weights(x, K, alpha_qp)


def _optimal_weights(x: np.ndarray, K: np.ndarray, alpha_qp: float = 1.0, eps=1e-9):
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
