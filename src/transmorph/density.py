#!/usr/bin/env python3

import numpy as np
from numba import njit, vectorize
from scipy import sparse
import osqp


@njit
def kernel_H(D, sigma):
    # Returns density entropy
    # vector is automatically normalized to sum up to 1.
    Dvector = _get_density(
        D,
        sigma*np.ones((D.shape[0],), dtype=np.float64),
        np.zeros((D.shape[0],), dtype=np.float64)
    ).sum(axis=1)
    Dvector /= Dvector.sum()
    return np.sum(Dvector * np.log(Dvector.shape[0] * Dvector))


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
    if v0 > v1 > v2: # backwards
        log_search = True
        step = 1/base
        v0, v2, s0, s2 = v2, v0, s2, s0
    elif v0 < v1 < v2: # forward
        log_search = True
        step = base

    # Log search
    for i in range(max_depth):
        if not log_search or v2 < v1:
            break
        v0, s0, v1, s1 = v1, s1, v2, s2
        s2 *= step
        v2 = kernel_H(D, s2)

    if s0 < s2:
        v0, v2, s0, s2 = v2, v0, s2, s0

    # Dichotomous search
    # s0 ---- m0 -- m1 ---- s2
    for i in range(max_depth):
        mid0 = s0 + 3*(s2 - s0)/8
        mid1 = mid0 + (s2 - s0) / 4
        v0 = kernel_H(D, mid0)
        v1 = kernel_H(D, mid1)
        if v0 > v1:
            s0, s2 = s0, mid1
        else:
            s0, s2 = mid0, s2
        if s2/s0 < thr:
            break

    return (s0 + s2)/2


def normal_kernel_weights(
        D: np.ndarray,
        scales: np.ndarray = None,
        offsets: np.ndarray = None,
        alpha_qp: float = 1.0
):
    ## Shortcut get_density -> optimal weights
    n = D.shape[0]
    if scales is None:
        scales = np.ones((n,), dtype=np.float32)
    if offsets is None:
        offsets = np.zeros((n,), dtype=np.float32)

    assert all(scales >= 0.0)
    assert all(offsets >= 0.0)
    assert scales.shape[0] == n
    assert offsets.shape[0] == n

    K = _get_density(D, scales, offsets)
    w = optimal_weights(K, alpha_qp)
    return w / np.sum(w)


@njit(fastmath=True)
def _get_density(
        D: np.ndarray,
        scales: np.ndarray,
        offsets: np.ndarray
) -> np.ndarray:
    n = D.shape[0]
    K = np.zeros((n, n), dtype=np.float32)
    inv_sqrt_2pi = 0.3989422804
    for i in range(n):
        sigma_i = scales[i]
        if sigma_i == 0.0:
            sigma_i = 1.0
        rho_i = offsets[i]
        normalizer = inv_sqrt_2pi / sigma_i
        for j in range(n):
            dij = (D[i,j] - rho_i)/sigma_i
            if dij < 0.0:
                dij = 0.0
            K[i, j] = np.exp(-dij**2/2)*normalizer
    return K


def optimal_weights(K: np.ndarray, alpha_qp: float = 1.0, eps=1e-12):
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
    x = np.clip(x, eps, 1.0)
    return x / x.sum()
