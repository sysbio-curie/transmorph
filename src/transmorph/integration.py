#!/usr/bin/env python3

import numpy as np
import ot

from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from .density import normal_kernel_weights
from .tdata import TData


def _transform(
        tdata_x: TData,
        transports: list,
        jitter: bool = True,
        jitter_std: float = .01,
        n_neighbors: int = 1) -> np.ndarray:
    """
    Optimal transport integration, inspired by (Ferradans 2013).

    Returns:
    --------
    N @ (P @ diag(1 /. wy)) @ yt,
    with N the row-normalizer N = diag( (P @ diag(1 /. wy)) @ 1 )^-1

    Parameters:
    -----------
    TODO: docstring
    """
    xs_raw = tdata_x.x_raw
    dx = np.zeros(xs_raw.shape)
    for n_iter, (tdata_xi, tdata_yi, Pxyi) in enumerate(transports):
        T = Pxyi.toarray() @ np.diag(1 / tdata_yi.weights())
        xti = np.diag(1 / T.sum(axis=1)) @ T @ tdata_yi.x_raw
        dxi = xti - tdata_xi.x_raw
        Di = tdata_x.distance(tdata_xi)
        idx = np.argsort(Di, axis=1)[:,:n_neighbors]
        idx_selector = np.zeros((len(xs_raw), len(xti)))
        np.put_along_axis(idx_selector, idx, 1/n_neighbors, axis=1)
        dx += idx_selector @ dxi
    x_int = xs_raw + dx / len(transports)

    if jitter:
        stdev = jitter_std * (np.max(x_int, axis=0) - np.min(x_int, axis=0))
        x_int = x_int + np.random.randn(*x_int.shape) * stdev
    return x_int


def _compute_transport(
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
    n, m = len(wx), len(wy)

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

