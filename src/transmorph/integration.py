#!/usr/bin/env python3

import numpy as np
import ot

from scipy.sparse import csr_matrix

from .density import normal_kernel_weights


def _transform(
        wy: np.ndarray,
        yt: np.ndarray,
        P: np.ndarray,
        jitter: bool = True,
        jitter_std: float = .01) -> np.ndarray:
    """
    Optimal transport integration (Ferradans 2013)

    Returns:
    --------
    N @ (P @ diag(1 /. wy)) @ yt,
    with N the row-normalizer N = diag( (P @ diag(1 /. wy)) @ 1 )^-1

    Parameters:
    -----------
    wy: (n,1) np.ndarray
        Reference dataset weights used during OT
    yt: (m,d) np.ndarray
        Target distribution points
    P:  (n,m) np.ndarray
        Optimal transport plan
    """
    T = P.toarray() @ np.diag(1 / wy)
    xt = np.diag(1 / T.sum(axis=1)) @ T @ yt
    if jitter:
        stdev = jitter_std * (np.max(xt, axis=0) - np.min(xt, axis=0))
        xt = xt + np.random.randn(*xt.shape) * stdev
    return xt


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
        hreg: float = 1e-3) -> np.ndarray:
    """
    Returns the optimal transport plan between xs and yt, interfaces the
    POT methods: https://github.com/PythonOT/POT

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
        Cost matrix, M_ij = cost(xi, yj).
    Mx: array (n,n)
        Cost matrix, M_ij = cost(xi, xj).
    My: array (m,m)
        Cost matrix, M_ij = cost(yi, yj).
    max_iter: int
        Maximum number of iterations for the OT solver.
    entropy: bool
        Use the Sinkhorn method with entropy regularization
    hreg: float
        Entropy regularizer for Sinkhorn's solver.
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

        assert Mxy is not None, "No cost matrix provided."
        assert Mxy.shape == (n, m), "Incompatible cost matrix.\
            Expected (%i,%i), found (%i,%i)." % (n, m, *Mxy.shape)

        Mxy /= Mxy.max()
        if entropy:
            transport_plan = ot.sinkhorn(wx, wy, Mxy, hreg, numItermax=max_iter)
        else:
            transport_plan = ot.emd(wx, wy, Mxy, numItermax=max_iter)

    if method == 'gromov':

        assert Mx is not None, "No cost matrix provided for xs."
        assert Mx.shape == (n, n), "Incompatible cost matrix.\
            Expected (%i,%i), found (%i,%i)." % (n, n, *Mx.shape)
        Mx /= Mx.max()

        assert My is not None, "No cost matrix provided for yt."
        assert Mx.shape == (m, m), "Incompatible cost matrix.\
            Expected (%i,%i), found (%i,%i)." % (m, m, *My.shape)
        My /= My.max()

        if entropy:
            transport_plan = ot.gromov.entropic_gromov_wasserstein(Mx, My, wx, wy, 'square_loss', hreg, numItermax=max_iter)
        else:
            transport_plan = ot.gromov.gromov_wasserstein(Mx, My, wx, wy, 'square_loss', numItermax=max_iter)

    return csr_matrix(transport_plan)
