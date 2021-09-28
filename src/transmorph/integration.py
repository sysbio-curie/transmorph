#!/usr/bin/env python3

import numpy as np
import ot

from numba import njit
from numpy.random import RandomState
from scipy.sparse import csr_matrix
from sklearn.utils import check_array
from umap.umap_ import (
    find_ab_params,
    simplicial_set_embedding
)

from .tdata import TData
from .constants import *
from .utils import (
    combine_graphs,
    compute_umap_graph,
    within_modality_stabilize
)


default_umap = {
    "n_components": 2,
    "initial_alpha": 1.0,
    "repulsion_strength": 1.5,
    "negative_sample_rate": 5.0,
    "n_epochs": 500,
    "spread": 1.0,
    "min_dist": 0.5,
    "densmap_keywords": {}
}


@njit
def project(P, y, yw):
    T = P @ np.diag(1 / yw)
    return np.diag(1 / T.sum(axis=1)) @ T @ y


def transform_latent_space(
        G1,
        G2,
        G12,
        G21,
        x_anchors,
        y_anchors,
        latent_dim: int = 2,
        umap_kwargs={},
        n_neighbors=15,
        metric="euclidean",
        metric_keywords={},
        random_state=42,
):
    # Unified graph representation
    A = combine_graphs(G1, G2, G12, G21, x_anchors, y_anchors, n_neighbors)
    
    # Embedding
    umap_params = {
        k: umap_kwargs.get(k, default_umap[k])
        for k in default_umap
    }
    a, b = find_ab_params(umap_params["spread"], umap_params["min_dist"])
    integrated_umap, _ = simplicial_set_embedding(
        None,
        A,
        umap_params["n_components"],
        umap_params["initial_alpha"],
        a,
        b,
        umap_params["repulsion_strength"],
        umap_params["negative_sample_rate"],
        umap_params["n_epochs"],
        "spectral",
        RandomState(random_state),
        metric,
        metric_keywords,
        False,
        umap_params["densmap_keywords"],
        False
    )
    return integrated_umap


def transform_reference_space(
        Pxy,
        X,
        x_weights,
        x_anchors,
        x_mapping,
        Y,
        y_weights,
        y_anchors,
        y_mapping,
        continuous_displacement=True
) -> np.ndarray:
    """
    Optimal transport integration, inspired by (Ferradans 2013).

    Parameters:
    -----------
    Pxy: (n0, m0) np.ndarray
        Optimal transport matrix between X and Y anchors, that marginalizes
        horizontally to x_weights[x_anchors] and vertically to
        y_weights[y_anchors].

    X: (n, d_X) np.ndarray
        Query dataset, containing n0 anchors.

    x_weights: (n,) np.ndarray
        Source distribution weights. If x_anchors[i] = 0, then x_weights[i] = 0.

    x_anchors: (n,) np.ndarray
        x_anchors[i] = True if i is an anchor of X.

    x_mapping: (n,)
        x_anchors[i] is the node index of X[i] anchor.

    Y: (n, d_Y) np.ndarray
        Reference dataset, containing m0 anchors.

    y_weights: (n,) np.ndarray
        Source distribution weights. If y_anchors[i] = 0, then y_weights[i] = 0.

    y_anchors: (n,) np.ndarray
        y_anchors[i] = True if i is an anchor of Y.

    y_mapping: (n,)
        y_anchors[i] is the node index of Y[i] anchor.

    Returns:
    --------
    N @ (P @ diag(1 /. wy)) @ yt,
    with N the row-normalizer N = diag( (P @ diag(1 /. wy)) @ 1 )^-1
    """

    n, m = X.shape[0], Y.shape[0]
    assert x_weights.shape[0] == n, "Source weights: wrong size."
    assert x_anchors.shape[0] == n, "Source anchors: wrong size."
    assert x_mapping.shape[0] == n, "Source mapping: wrong size."
    assert y_weights.shape[0] == m, "Reference weights: wrong size."
    assert y_anchors.shape[0] == m, "Reference anchors: wrong size."
    assert y_mapping.shape[0] == m, "Reference mapping: wrong size."

    # Delegate to njit
    return _transform_reference_space(
        check_array(Pxy.toarray(), dtype=np.float32, order="C"),
        check_array(X, dtype=np.float32, order="C"),
        np.float32(x_weights),
        x_anchors,
        x_mapping,
        check_array(Y, dtype=np.float32, order="C"),
        np.float32(y_weights),
        y_anchors,
        continuous_displacement=continuous_displacement
    )


@njit
def _transform_reference_space(
        Pxy,
        X,
        xw,
        x_anchors_sel,
        x_mapping,
        Y,
        yw,
        y_anchors_sel,
        continuous_displacement=True
):
    Y_anchors = Y[y_anchors_sel]
    sel_x, sel_y = xw > 0, yw > 0

    # Eliminating zero-weighted points
    (
        yw_nz,
        Y_anchors_nz,
        Pxy_nz
    ) = (
        yw[sel_y],
        Y_anchors[sel_y],
        Pxy[sel_x][:,sel_y]
    )
    X_anchors_projected = project(Pxy_nz, Y_anchors_nz, yw_nz)
    if not continuous_displacement:
        return X_anchors_projected

    X_anchors = X[x_anchors_sel]
    X_anchors_int = X_anchors.copy()
    X_anchors_int[sel_x] = X_anchors_projected
    delta_int = X_anchors_int - X_anchors
    small_idx = np.array([
        np.sum(x_anchors_sel[:x_mapping[i]])
        for i in range(len(X))
    ])
    return X + delta_int[small_idx]


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
    mreg: float = 1e-3
) -> np.ndarray:
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
    ), "Source weights must be in the probability simplex. "\
    f"Min: {np.min(wy)}, Sum: {np.sum(wy)}"
    assert abs(np.sum(wy) - 1) < 1e-9 and all(
        wy >= 0
    ), "Reference weights must be in the probability simplex. "\
    f"Min: {np.min(wy)}, Sum: {np.sum(wy)}"

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
            transport_plan = ot.unbalanced.sinkhorn_stabilized_unbalanced(wx, wy, Mxy, hreg, mreg, numItermax=max_iter)
        elif entropy:
            transport_plan = ot.bregman.sinkhorn_stabilized(wx, wy, Mxy, hreg, numItermax=max_iter)
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
            transport_plan = ot.gromov.entropic_gromov_wasserstein(Mx, My, wx, wy, 'square_loss', hreg, max_iter=max_iter)
        else:
            transport_plan = ot.gromov.gromov_wasserstein(Mx, My, wx, wy, 'square_loss', numItermax=max_iter)

    # TODO: optimize this inefficient block
    if slicing:
        tmp_transport_plan = transport_plan
        transport_plan = np.zeros((n, m))
        for i, j in enumerate(sel_x):
            transport_plan[j,sel_y] = tmp_transport_plan[i,:]

    return csr_matrix(transport_plan)

