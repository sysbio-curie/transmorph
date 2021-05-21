#!/usr/bin/env python3

import numpy as np
import ot

from typing import Callable

from .density import normal_kernel_weights


def ot_integration(
    xs: np.ndarray,
    yt: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
    M: np.ndarray = None,
    max_iter: int = 1e7,
    solver: str = 'sinkhorn',
    hreg: float = 1e-3,
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
    M: array (n,m)
        Cost matrix, M_ij = cost(xi, yj). If null, Euclidean distance by default.
    max_iter: int
        Maximum number of iterations for the OT solver.
    solver: str
        Belongs to {'emd', 'sinkhorn'}. Choose the exact/approximate solver.
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
    if M is None:
        M = ot.dist(xs, yt)
    assert M.shape == (n, m), "Cost function should return a pairwise (n,m) matrix."
    M /= M.max()

    # Normalization of weights
    assert abs(np.sum(wx) - 1) < 1e-9 and all(
        wx >= 0
    ), "Source weights must be in the probability simplex."
    assert abs(np.sum(wy) - 1) < 1e-9 and all(
        wy >= 0
    ), "Reference weights must be in the probability simplex."

    if verbose:
        print("WOTi > Computing optimal transport plan...")

    if solver == 'emd':
        transport_plan = ot.emd(wx, wy, M, numItermax=max_iter)
    elif solver == 'sinkhorn':
        transport_plan = ot.sinkhorn(wx, wy, M, hreg, numItermax=max_iter)
    else:
        print("Unrecognized solver: %s (valid are 'emd', 'sinkhorn')" % solver)
        raise ValueError

    if verbose:
        print("WOTi > Projecting source dataset...")

    # Casting to an array to ensure no ArrayView is returned.
    return np.array(np.diag(1 / wx) @ transport_plan @ yt)


def ot_transform(
    xs: np.ndarray,
    yt: np.ndarray,
    M: np.ndarray = None,
    max_iter: int = 1e7,
    solver: str = 'sinkhorn',
    hreg: float = 1e-3,
    weighted: bool = True,
    scale_src: float = 1,
    scale_ref: float = 1,
    alpha_qp: float = 1.0,
    verbose: bool = False,
):
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
        Belongs to {'emd', 'sinkhorn'}. Choose the exact/approximate solver.
    hreg: float
        Entropy regularizer for Sinkhorn's solver.
    weighted: bool
        Use the unsupervised weight selection
    scale_src: float
       Standard deviation of the Gaussian kernel used for source density correction.
    scale_ref: float
       Standard deviation of the Gaussian kernel used for target density correction.
    alpha_qp: float
        Parameter to provide to the quadratic program solver.
    verbose: bool
        Displays information in the standard output.
    """
    # Computing equal weights
    n, m = len(xs), len(yt)
    assert n >= 0, "Source matrix cannot be empty."
    assert m >= 0, "Reference matrix cannot be empty."

    if not weighted:
        wx, wy = np.array([1 / n] * n), np.array([1 / m] * m)
    else:
        if verbose:
            print("WOTi > Computing source distribution weights...")
        wx = normal_kernel_weights(
            xs, scale=scale_src, alpha_qp=alpha_qp
        )
        if verbose:
            print("WOTi > Computing reference distribution weights...")
        wy = normal_kernel_weights(
            yt, scale=scale_ref, alpha_qp=alpha_qp
        )

    # Adjusting for approximation error
    wx /= np.sum(wx)
    wy /= np.sum(wy)

    return ot_integration(
        xs, yt, wx, wy, M=M, max_iter=max_iter, solver=solver, hreg=hreg, verbose=verbose
    )
