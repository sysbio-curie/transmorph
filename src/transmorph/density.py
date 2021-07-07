#!/usr/bin/env python3

import numpy as np
from numba import njit, vectorize
from scipy import sparse
import osqp


@njit
def kernel_H(D, sigma):
    # Returns density entropy
    # vector is automatically normalized to sum up to 1.
    Dvector = get_density(D, sigma).sum(axis=1)
    Dvector /= Dvector.sum()
    return -np.sum(Dvector * np.log(Dvector)/np.log(2))


@njit
def sigma_search(D, max_depth=20, base=2, thr=1.01):
    # Searches for a sigma that minimizes density entropy
    # for a given distance matrix D
    # Step 1) Log search
    # Step 2) Dichotomous search

    # Initialization
    s0, s1, s2 = 1/base, 1, base
    v0 = kernel_H(D, s0)
    v1 = kernel_H(D, s1)
    v2 = kernel_H(D, s2)

    # Choosing direction of the minimum
    log_search = False
    step = 0
    if v0 < v1 < v2: # _/, backwards
        log_search = True
        step = 1/base
        v0, v2, s0, s2 = v2, v0, s2, s0
    elif v0 > v1 > v2: # \_, forward
        log_search = True
        step = base

    # Log search
    for i in range(max_depth):
        if not log_search or v2 > v1:
            break
        v0, s0, v1, s1 = v1, s1, v2, s2
        s2 *= step
        v2 = kernel_H(D, s2)

    if s0 > s2:
        v0, v2, s0, s2 = v2, v0, s2, s0

    # Dichotomous search
    # s0 ---- m0 -- m1 ---- s2
    for i in range(max_depth):
        mid0 = s0 + 3*(s2 - s0)/8
        mid1 = mid0 + (s2 - s0) / 4
        v0 = kernel_H(D, mid0)
        v1 = kernel_H(D, mid1)
        if v0 < v1:
            s0, s2 = s0, mid1
        else:
            s0, s2 = mid0, s2
        if s2/s0 < thr:
            break

    return (s0 + s2)/2


def normal_kernel_weights(
        D: np.ndarray, scale: float = 1, alpha_qp: float = 1.0
):
    ## Shortcut get_density -> optimal weights
    assert scale > 0, "scale must be positive, found %f" % scale
    K = get_density(D, scale)
    w = optimal_weights(K, alpha_qp)
    return w / np.sum(w)


@vectorize
def get_density(D: np.ndarray, scale: float = 1) -> np.ndarray:
    return np.exp(-D**2/(2*scale))/(2.5066282746310002*scale)


def optimal_weights(K: np.ndarray, alpha_qp: float = 1.0, eps=1e-9):
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
