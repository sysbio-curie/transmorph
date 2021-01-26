#!/usr/bin/env python3

import numpy as np
from cvxopt import matrix, solvers
from awkde import GaussianKDE
from scipy.stats import multivariate_normal
import math


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


def normal_kernel_weights(x: np.ndarray, scale: float = 1, alpha: float = 0.5):

    assert scale > 0, "sigma must be positive."
    assert 0 < alpha < 1, "alpha must be in [0,1]"

    K = _gaussian_kernel_matrix(x, scale, alpha)
    return _optimal_weights(x, K)


def _optimal_weights(x: np.ndarray, K: np.ndarray):
    """ Computes optimal weights given K pairwise kernel matrix. """

    # Cost matrix
    M_np = K.mean(axis=1)
    KM_np = K - M_np
    P_np = np.dot(KM_np.T, KM_np)
    n = len(P_np)
    P = matrix(P_np, (n, n))
    p = matrix([1.0 / n] * n, (n, 1))  # alpha vector

    # Inequality constraints
    G = matrix(np.vstack((np.diag([-1.0] * n), KM_np)), (2 * n, n))
    h = matrix([0.0] * n + [1 / (n * math.sqrt(2 * math.pi))] * n, (2 * n, 1))

    # Equality constraints
    A = matrix([1.0] * n, (1, n))
    b = matrix(1.0, (1, 1))

    sol = solvers.qp(P, p, G, h, A, b, verbose=False)
    return np.array(sol["x"]).reshape(-1)
