#!/usr/bin/env python3

import numpy as np
import ot

from typing import Callable

from .density import normal_kernel_weights

def gw_integration(
    xs: np.ndarray,
    yt: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
    Mx: np.ndarray = None,
    My: np.ndarray = None,
    max_iter: int = 1000,
    solver: str = 'gw_entropy',
    hreg: float = 1e-4,
    verbose: bool = False,
) -> np.ndarray:
    """
    Optimal transport-based dataset integration.

    Returns:
    -------
    np.ndarray (n,d) -- Projection of $xs onto $yt.

    Parameters:
    ----------
    xs: array (n,d)
        Source disribution points.
    yt: array (m,d)
        Target distribution points.
    wx: array (n,1)
        Source weights histogram (sum to 1).
    wy: array (m,1)
        Target weights histogram (sum to 1).
    Mx: array (n,n)
        Cost matrix, M_ij = cost(xi, xj). If null, Euclidean distance by default.
    My: array (m,m)
        Cost matrix, M_ij = cost(yi, yj). If null, Euclidean distance by default.
    max_iter: int
        Maximum number of iterations for the OT solver.
    solver: str
        Belongs to {'gw', 'sinkhorn'}. Choose the exact/approximate solver.
    hreg: float
        Entropy regularizer for Sinkhorn's solver.
    verbose: bool
        Displays information in the standard output.
    """
    n, m = len(xs), len(yt)
    assert n >= 0, "Source matrix cannot be empty."
    assert m >= 0, "Reference matrix cannot be empty."
    assert n == len(wx), "Weights size does not coincidate with source points."
    assert m == len(wy), "Weights size does not coincidate with reference points."

    # Computing weights
    if Mx is None:
        Mx = ot.dist(xs, xs)
    if My is None:
        My = ot.dist(yt, yt)
    assert Mx.shape == (n, n), "Cost function should return a pairwise (n,m) matrix."
    assert My.shape == (m, m), "Cost function should return a pairwise (n,m) matrix."
    Mx /= Mx.max()
    My /= My.max()

    # Normalized weights
    assert abs(np.sum(wx) - 1) < 1e-9 and all(
        wx >= 0
    ), "Source weights must be in the probability simplex."
    assert abs(np.sum(wy) - 1) < 1e-9 and all(
        wy >= 0
    ), "Reference weights must be in the probability simplex."

    if verbose:
        print("WOTi > Computing Gromov-Wasserstein plan...")
    if solver == 'gw':
        transport_plan = ot.gromov.gromov_wasserstein(Mx, My, wx, wy, 'square_loss', numItermax=max_iter)
    elif solver == 'gw_entropy':
        transport_plan = ot.gromov.entropic_gromov_wasserstein(Mx, My, wx, wy, 'square_loss', hreg, numItermax=max_iter)
    else:
        print("Unrecognized solver: %s (valid are 'gw', 'gw_entropy')" % solver)
        raise ValueError

    if verbose:
        print("WOTi > Projecting source dataset...")
    return np.array(np.diag(1 / wx) @ transport_plan @ yt)


def gw_transform(
    xs: np.ndarray,
    yt: np.ndarray,
    Mx: np.ndarray = None,
    My: np.ndarray = None,
    max_iter: int = 1000,
    solver: str = 'gw_entropy',
    hreg: float = 1e-4,
    weighted: bool = True,
    alpha_qp: float = 1.0,
    scale: float = 1.0,
    verbose: bool = False,
) -> np.ndarray:
    """
    Optimal transport dataset integration.

    Returns:
    -------
    np.ndarray (n,d) -- Projection of $xs onto $yt.

    Parameters:
    ----------
    xs: array (n,d)
        Source disribution points.
    yt: array (m,d)
        Target distribution points.
    wx: array (n,1)
        Source weights histogram (sum to 1).
    wy: array (m,1)
        Target weights histogram (sum to 1).
    M: array (n,m)
        Cost matrix, M_ij = cost(xi, yj). If null, Euclidean disance by default.
    max_iter: int
        Maximum number of iterations for the OT solver.
    solver: str
        Belongs to {'gw', 'sinkhorn'}. Choose the exact/approximate solver.
    hreg: float
        Entropy regularizer for Sinkhorn's solver.
    weighted: bool
        Use the unsupervised weight selection
    alpha_qp: float
        Parameter to provide to the quadratic program solver.
    verbose: bool
        Displays information in the standard output.
    """
    # Computing equal weights
    n, m = len(xs), len(yt)
    assert n >= 0, "Source matrix cannot be empty."
    assert m >= 0, "Reference matrix cannot be empty."
    assert scale > 0, "Scale must be positive."
    if weighted:
        if verbose:
            print("WOTi > Computing source distribution weights...")
        wx = normal_kernel_weights(xs, alpha_qp=alpha_qp, scale=scale)
        if verbose:
            print("WOTi > Computing reference distribution weights...")
        wy = normal_kernel_weights(yt, alpha_qp=alpha_qp, scale=scale)
    else:
        wx, wy = np.array([1 / n] * n), np.array([1 / m] * m)

    # Adjusting for approximation
    wx /= np.sum(wx)
    wy /= np.sum(wy)

    return gw_integration(
        xs, yt, wx, wy, Mx, My, max_iter=max_iter, solver=solver, verbose=verbose
    )
