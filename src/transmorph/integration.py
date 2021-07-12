#!/usr/bin/env python3

import numpy as np
import ot

from numba import njit
from scipy.sparse import csr_matrix

from .tdata import TData
from .constants import *

def transform(
        transport,
        jitter: bool = True,
        jitter_std: float = .01) -> np.ndarray:
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
    tdata_x, tdata_y, Pxy = transport
    return _transform(
        Pxy.toarray(),
        tdata_x.X,
        tdata_x.weights(),
        tdata_x.anchors,
        tdata_x.anchors_map,
        tdata_y.X,
        tdata_y.weights(),
        tdata_y.anchors,
        jitter,
        jitter_std
    )

def _transform(Pxy,
               x,
               xw,
               x_anchors_sel,
               x_mapping,
               y,
               yw,
               y_anchors_sel,
               jitter,
               jitter_std):
    sel_x, sel_y = xw > 0, yw > 0
    x_anchors = x[x_anchors_sel]
    y_anchors = y[y_anchors_sel]
    yw_nz, y_anchors_nz, Pxy_nz = ( # Eliminating zero-weighted points
        yw[sel_y],
        y_anchors[sel_y],
        Pxy[sel_x][:,sel_y]
    )
    x_anchors_int = x_anchors.copy()
    T = Pxy_nz @ np.diag(1 / yw_nz)
    x_anchors_int[sel_x] = np.diag(1 / T.sum(axis=1)) @ T @ y_anchors_nz
    delta_int = x_anchors_int - x_anchors
    small_idx = [
        np.sum(x_anchors_sel[:x_mapping[i]])
        for i in range(len(x))
    ]
    x_int = x + delta_int[small_idx]

    if jitter:
        stdev = jitter_std * (np.max(x_int, axis=0) - np.min(x_int, axis=0))
        x_int = x_int + np.random.randn(*x_int.shape) * stdev

    return x_int


def compute_transport(
        wx: np.ndarray,
        wy: np.ndarray,
        method: int = TR_METHOD_OT,
        Mxy: np.ndarray = None,
        Mx: np.ndarray = None,
        My: np.ndarray = None,
        max_iter: int = 1e7,
        entropy: bool = False,
        hreg: float = 1e-3,
        unbalanced: bool = False,
        mreg: float = 1e-3) -> np.ndarray:
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
    method: int in TR_METHOD
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

    sel_x, sel_y = np.argwhere(wx != 0)[:,0], np.argwhere(wy != 0)[:,0]
    slicing = len(sel_x) < n or len(sel_y) < n
    if slicing:
        wx, wy = wx[sel_x], wy[sel_y]

    if method == TR_METHOD_OT:

        assert Mxy is not None, "No cost matrix provided."
        assert Mxy.shape == (n, m), "Incompatible cost matrix.\
            Expected (%i,%i), found (%i,%i)." % (n, m, *Mxy.shape)

        if slicing:
            Mxy = Mxy[sel_x][:,sel_y].copy()

        Mxy /= Mxy.max()
        if unbalanced:
            transport_plan = ot.sinkhorn_unbalanced(wx, wy, Mxy, hreg, mreg, numItermax=max_iter)
        elif entropy:
            transport_plan = ot.sinkhorn(wx, wy, Mxy, hreg, numItermax=max_iter)
        else:
            transport_plan = ot.emd(wx, wy, Mxy, numItermax=max_iter)

    if method == TR_METHOD_GROMOV:

        assert Mx is not None, "No cost matrix provided for xs."
        assert Mx.shape == (n, n), "Incompatible cost matrix.\
            Expected (%i,%i), found (%i,%i)." % (n, n, *Mx.shape)
        if slicing:
            Mx = Mx[sel_x][:,sel_x]
        Mx /= Mx.max()

        assert My is not None, "No cost matrix provided for yt."
        assert My.shape == (m, m), "Incompatible cost matrix.\
            Expected (%i,%i), found (%i,%i)." % (m, m, *My.shape)
        if slicing:
            My = My[sel_y][:,sel_y]
        My /= My.max()

        if unbalanced:
            raise NotImplementedError
        elif entropy:
            transport_plan = ot.gromov.entropic_gromov_wasserstein(Mx, My, wx, wy, 'square_loss', hreg, numItermax=max_iter)
        else:
            transport_plan = ot.gromov.gromov_wasserstein(Mx, My, wx, wy, 'square_loss', numItermax=max_iter)

    # TODO: clean this block
    if slicing:
        tmp_transport_plan = transport_plan
        transport_plan = np.zeros((n, m))
        for i, j in enumerate(sel_x):
            transport_plan[j,sel_y] = tmp_transport_plan[i,:]

    return csr_matrix(transport_plan)

