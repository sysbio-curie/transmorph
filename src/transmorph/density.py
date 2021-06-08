#!/usr/bin/env python3

import numpy as np
from scipy.stats import norm, entropy
from scipy.spatial.distance import cdist
from scipy import sparse
import osqp


def _kernel_var(xs, sigma):
    p = _get_density(xs, sigma).sum(axis=1)
    return entropy(p / p.sum())


def sigma_search(xs, max_depth=20, base=2, init_s=1, thr=1.01):
    # Searches for a sigma that maximizes density entropy

    # Initialization
    s0, s1, s2 = 1, 1/base, 1/(base*base)
    v0 = _kernel_var(xs, s0)
    v1 = _kernel_var(xs, s1)
    v2 = _kernel_var(xs, s2)

    # Log search
    for i in range(max_depth):
        if v2 > v1:
            break
        v0 = v1
        s0 = s1
        v1 = v2
        s1 = s2
        s2 /= base
        v2 = _kernel_var(xs, s2)

    # Trichotomous search
    for i in range(max_depth):
        mid0 = s0 + 3*(s2 - s0)/8
        mid1 = mid0 + (s2 - s0) / 4
        v0 = _kernel_var(xs, mid0)
        v1 = _kernel_var(xs, mid1)
        if v0 < v1:
            s0, s2 = s0, mid1
        else:
            s0, s2 = mid0, s2
        if s2/s0 < thr:
            break

    return (s0 + s2)/2


def normal_kernel_weights(
        x: np.ndarray, alpha_qp: float = 1.0, scale: float = 1
):
    ## Shortcut get_density -> optimal weights
    assert 0 < alpha_qp < 2, "alpha_qp must be in (0,2), found %f" % alpha_qp
    K = _get_density(x, scale)
    w = _optimal_weights(K, alpha_qp)
    return w / np.sum(w)


def _get_density(x: np.ndarray, scale: float = 1) -> np.ndarray:
    assert scale > 0, "scale must be positive, found %f" % scale
    std_arr = x.std(axis=0)
    std_arr[std_arr == 0] = 1 # Fixes 0-std columns
    xnorm = (x / std_arr)
    Dmatrix = cdist(xnorm, xnorm)
    assert Dmatrix.max() > 0, "All points are equal in x."
    return norm.pdf(-Dmatrix, loc=0, scale=scale)


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
