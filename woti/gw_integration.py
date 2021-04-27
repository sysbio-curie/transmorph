#!/usr/bin/env python3

import numpy as np
import ot

from typing import Callable

from .density import normal_kernel_weights


def _gw_transform(
    xs: np.ndarray,
    yt: np.ndarray,
    ws: np.ndarray,
    vs: np.ndarray,
    max_iter: int = 1e7,
    cost_function: Callable = ot.dist,
    verbose: bool = False,
) -> np.ndarray:
    """
    Gromov-Wasserstein-based dataset integration.

    Returns:
    -------
    np.ndarray (n,d) -- Projection of $xs onto $yt.

    Parameters:
    ----------
    xs: array (n,d)
        Source cloud point.
    yt: array (m,d)
        Reference cloud point.
    ws: array (n,1)
        Source cloud point weights for the discrete optimal transport
    vs: array (m,1)
        Reference cloud point weights for the discrete optimal transport
    max_iter: int
        Maximum number of iterations for the OT problem
    cost_function: Callable ( array (n,d), array(m,d) ) -> array (n,m)
        This function computes the paiwise cost matrix between xs and yt records.
        Default: Euclidean distance
    verbose: bool
        Displays information in the standard output.
    """
    n, m = len(xs), len(yt)
    assert n >= 0, "Source matrix cannot be empty."
    assert m >= 0, "Reference matrix cannot be empty."
    assert n == len(ws), "Weights size does not coincidate with source points."
    assert m == len(
        vs), "Weights size does not coincidate with reference points."

    # Computing weights
    Ms = cost_function(xs, xs)
    assert Ms.shape == (
        n, n), "Cost function should return a pairwise (n,m) matrix."
    Ms /= Ms.max()

    Mt = cost_function(yt, yt)
    assert Mt.shape == (
        m, m), "Cost function should return a pairwise (n,m) matrix."
    Mt /= Mt.max()

    # Normalization of weights
    assert abs(np.sum(ws) - 1) < 1e-9 and all(
        ws >= 0
    ), "Source weights must be in the probability simplex."
    assert abs(np.sum(vs) - 1) < 1e-9 and all(
        vs >= 0
    ), "Reference weights must be in the probability simplex."

    if verbose:
        print("WOTi > Computing Gromov-Wasserstein plan...")
    transport_plan = ot.gromov.gromov_wasserstein(Ms, Mt, ws, vs, 'square_loss', numItermax=max_iter)

    if verbose:
        print("WOTi > Projecting source dataset...")
    return np.diag(1 / ws) @ transport_plan @ yt


def bgw_transform(
    xs: np.ndarray,
    yt: np.ndarray,
    max_iter: int = 1e7,
    cost_function: Callable = ot.dist,
    verbose: bool = False,
):
    """
    Balanced Gromov-Wasserstein dataset integration (equal weights).

    Returns:
    -------
    np.ndarray (n,d) -- Projection of $xs onto $yt.

    Parameters:
    ----------
    xs: array (n,d)
        Source cloud point.
    yt: array (m,d)
        Reference cloud point.
    max_iter: int
        Maximum number of iterations for the OT problem
    cost_function: Callable ( array (n,d), array(m,d) ) -> array (n,m)
        This function computes the paiwise cost matrix between xs and yt records.
        Default: Euclidean distance
    verbose: bool
        Displays information in the standard output.
    """
    # Computing equal weights
    n, m = len(xs), len(yt)
    assert n >= 0, "Source matrix cannot be empty."
    assert m >= 0, "Reference matrix cannot be empty."
    ws, vs = np.array([1 / n] * n), np.array([1 / m] * m)

    # Adjusting for approximation
    ws /= np.sum(ws)
    vs /= np.sum(vs)

    return _gw_transform(
        xs, yt, ws, vs, max_iter=max_iter, cost_function=cost_function, verbose=verbose
    )


def gw_transform(
    xs: np.ndarray,
    yt: np.ndarray,
    max_iter: int = 1e7,
    cost_function: Callable = ot.dist,
    scale: float = 1,
    scale_ref: float = -1,
    alpha_qp: float = 1.0,
    verbose: bool = False,
):
    """
    Gromov-Wasserstein-based dataset integration.

    Returns:
    -------
    np.ndarray (n,d) -- Projection of $xs onto $yt.

    Parameters:
    ----------
    xs: array (n,d)
        Source cloud point.
    yt: array (m,d)
        Reference cloud point.
    ws: array (n,1)
        Source cloud point weights for the discrete optimal transport
    vs: array (m,1)
        Reference cloud point weights for the discrete optimal transport
    max_iter: int
        Maximum number of iterations for the OT problem
    cost_function: Callable ( array (n,d), array(m,d) ) -> array (n,m)
        This function computes the paiwise cost matrix between xs and yt records.
        Default: Euclidean distance
    scale: float
        Kernels scaling for density correction. If $scale_ref = -1, $scale affects
        both source and reference kernels. Otherwise, it affects only source distribution.
    scale_ref: float
        Kernels scaling for density correction, affects reference distribution.
    alpha_kde: float
        Alpha parameter for KDE bandwith selection, between 0 and 1.
    alpha_qp: float
        Alpha parameter for OSQP solver, between 0 and 2.
    verbose: bool
        Displays information in the standard output.
    """

    if scale_ref == -1:
        scale_ref = scale

    n, m = len(xs), len(yt)
    assert n >= 0, "Source matrix cannot be empty."
    assert m >= 0, "Reference matrix cannot be empty."
    if verbose:
        print("WOTi > Computing source distribution weights...")
    ws = normal_kernel_weights(
        xs, scale=scale, alpha_qp=alpha_qp)
    if verbose:
        print("WOTi > Computing reference distribution weights...")
    vs = normal_kernel_weights(
        yt, scale=scale_ref, alpha_qp=alpha_qp
    )

    return _gw_transform(
        xs, yt, ws, vs, max_iter=max_iter, cost_function=cost_function, verbose=verbose
    )
