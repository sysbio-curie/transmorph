#!/usr/bin/env python3

import numpy as np
import ot

from typing import Callable
from scipy.sparse import csr_matrix

from .density import normal_kernel_weights

def _transform(wx: np.ndarray, yt: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Optimal transport integration (Ferradans 2013)

    Returns:
    --------
    diag(1 /. wx) @ P @ yt

    Parameters:
    -----------
    wx: (n,1) np.ndarray
        Optimal transport weights
    yt: (m,d) np.ndarray
        Target distribution
    P:  (n,m) np.ndarray
        Optimal transport plan
    """
    n = wx.shape[0]
    m = yt.shape[0]
    assert P.shape == (n,m), "Dimension mismatch, (%i,%i) != (%i,%i)" % (
        *P.shape, n, m
    )
    return np.array(np.diag(1 / wx) @ P @ yt)


def _compute_transport(
    xs: np.ndarray,
    yt: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
    method: str = 'ot',
    Mxy: np.ndarray = None,
    Mx: np.ndarray = None,
    My: np.ndarray = None,
    max_iter: int = 1e7,
    entropy: bool = False,
    hreg: float = 1e-3,
    verbose: bool = False,
) -> np.ndarray:
    """
    Returns the ptimal transport plan between xs and yt

    Returns:
    -------
    csr_matrix (n,d) -- Projection of $xs onto $yt.

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
    method: str in {'ot', 'gromov'}
        Optimal transport or Gromov-Wasserstein integration
    Mxy: array (n,m)
        Cost matrix, M_ij = cost(xi, yj). If null, Euclidean distance by default.
    Mx: array (n,n)
        Cost matrix, M_ij = cost(xi, xj). If null, Euclidean distance by default.
    My: array (m,m)
        Cost matrix, M_ij = cost(yi, yj). If null, Euclidean distance by default.
    max_iter: int
        Maximum number of iterations for the OT solver.
    entropy: bool
        Use the Sinkhorn method with entropy regularization
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

    # Normalization of weights
    assert abs(np.sum(wx) - 1) < 1e-9 and all(
        wx >= 0
    ), "Source weights must be in the probability simplex."
    assert abs(np.sum(wy) - 1) < 1e-9 and all(
        wy >= 0
    ), "Reference weights must be in the probability simplex."

    if method == 'ot':

        if Mxy is None:
            assert xs.shape[1] == yt.shape[1], "Dimensionality error.\
                xs has shape (%i,%i) and yt has shape (%i,%i), with no cost matrix\
                provided. Impossible to use Euclidean distance." % (*xs.shape, *yt.shape)
            Mxy = ot.dist(xs, yt)
        assert Mxy.shape == (n, m), "Incompatible cost matrix.\
            Expected (%i,%i), found (%i,%i)." % (n, m, *Mxy.shape)
        Mxy /= Mxy.max()
        if verbose:
            print("WOTi > Computing optimal transport plan...")
        if entropy:
            transport_plan = ot.sinkhorn(wx, wy, Mxy, hreg, numItermax=max_iter)
        else:
            transport_plan = ot.emd(wx, wy, Mxy, numItermax=max_iter)

    if method == 'gromov':

        if Mx is None:
            Mx = ot.dist(xs, xs)
        assert Mx.shape == (n, n), "Incompatible cost matrix.\
            Expected (%i,%i), found (%i,%i)." % (n, n, *Mx.shape)
        Mx /= Mx.max()
        if My is None:
            My = ot.dist(yt, yt)
        assert Mx.shape == (m, m), "Incompatible cost matrix.\
            Expected (%i,%i), found (%i,%i)." % (m, m, *My.shape)
        My /= My.max()

        if verbose:
            print("WOTi > Computing Gromov-Wasserstein plan...")
        if entropy:
            transport_plan = ot.gromov.entropic_gromov_wasserstein(Mx, My, wx, wy, 'square_loss', hreg, numItermax=max_iter)
        else:
            transport_plan = ot.gromov.gromov_wasserstein(Mx, My, wx, wy, 'square_loss', numItermax=max_iter)

    return csr_matrix(transport_plan)

