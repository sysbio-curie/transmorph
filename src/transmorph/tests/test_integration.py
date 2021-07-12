#!/usr/bin/env python3

import pytest
import numpy as np

from scipy.spatial.distance import cdist
from sklearn.utils._testing import assert_array_almost_equal

from ..constants import *
from ..datasets import load_cell_cycle
from ..datasets import load_spirals
from ..tdata import TData
from ..transmorph import Transmorph
from ..transmorph import _aliases_methods

# For now, tests will not challenge the return value
# They will rather test for code inconsistencies
# TODO: Think of testing metrics


def test_transmorph_validate_parameters():

    # methods
    t = Transmorph(method='ot')
    assert t.method == TR_METHOD_OT
    t = Transmorph(method='gromov')
    assert t.method == TR_METHOD_GROMOV

    with pytest.raises(AssertionError):
        Transmorph(method='foo')

    # aliases
    for strategy in [
            'woti',
            'auto',
            'automatic',
            'qp',
            'labeled',
            'labels',
            'uniform',
            'none',
            'nil',
            'uni']:
        t = Transmorph(weighting_strategy=strategy)
        assert t.weighting_strategy == _aliases_methods[strategy]

    with pytest.raises(AssertionError):
        Transmorph(weighting_strategy='foo')

    # label dependency
    Transmorph(label_dependency=0)
    Transmorph(label_dependency=0.5)
    Transmorph(label_dependency=1)

    with pytest.raises(AssertionError):
        Transmorph(label_dependency=-.1)
    with pytest.raises(AssertionError):
        Transmorph(label_dependency=1.1)

    # max_iter
    Transmorph(max_iter=100)
    with pytest.raises(AssertionError):
        Transmorph(max_iter=-1)

    # combinations
    with pytest.raises(AssertionError):
        Transmorph(method='gromov', weighting_strategy='labels')

    with pytest.raises(AssertionError):
        Transmorph(method='gromov', unbalanced=True)

    with pytest.raises(AssertionError):
        Transmorph(method='ot', geodesic=True)


def test_tdata_validate_parameters():
    xs, _ = load_spirals()
    n = len(xs)
    weights = np.ones(n) / n
    weights_n = np.ones(n) # should be corrected automatically to sum up to 1
    labels = np.ones(n)
    weights_w = np.ones(n+1) / (n+1)
    labels_w = np.ones(n+1)
    t = TData(xs)
    assert n == len(t)
    assert t.layers['raw'] is t.X
    t = TData(xs, weights)
    t = TData(xs, weights_n)
    assert abs(t.weights().sum() - 1) < 1e-6
    t = TData(xs, labels)

    with pytest.raises(AssertionError):
        TData(np.ndarray([]))
    with pytest.raises(AssertionError):
        TData(xs, weights_w)
    with pytest.raises(AssertionError):
        TData(xs, labels_w)


def test_tdata_pca_3d_2d():
    xs, xt = load_spirals()
    ys, _ = load_cell_cycle()
    txs = TData(xs)
    txt = TData(xt)
    txs.pca(n_components=2)
    txs.pca(n_components=2, other=txt)
    with pytest.raises(AssertionError):
        txs.pca(n_components=2, other=TData(ys))


def test_tdata_neighbors():
    xs, _ = load_spirals()
    t = TData(xs)
    t.neighbors()
    with pytest.raises(AssertionError):
        t.neighbors(layer='pca')
    t.pca(n_components=2)
    t.neighbors(layer='pca')


def test_tdata_representers():
    xs, _ = load_spirals()
    t = TData(xs)
    with pytest.raises(AssertionError):
        t.select_representers()
    t.neighbors()
    t.select_representers()


def test_tdata_distance():
    xs, yt = load_spirals()
    tx, ty = TData(xs, normalize=False), TData(yt, normalize=False)
    D1t = cdist(xs, yt, metric='sqeuclidean')
    D1 = tx.distance(ty, metric='sqeuclidean')
    D2t = cdist(xs, xs, metric='sqeuclidean')
    D2 = tx.distance(metric='sqeuclidean')
    with pytest.raises(AssertionError):
        tx.distance(layer='pca')
    tx.pca(n_components=2)
    tx.distance(layer='pca')
    with pytest.raises(AssertionError):
        tx.distance(ty, layer='pca')


def test_tdata_compute_weights_uniform():
    xs, _ = load_spirals()
    tx = TData(xs)
    tx.compute_weights(method=TR_WS_UNIFORM)
    assert_array_almost_equal(tx.weights(), np.ones(len(xs)) / len(xs))


def test_tdata_compute_weights_woti():
    xs, _ = load_spirals()
    tx = TData(xs)
    tx.compute_weights(method=TR_WS_AUTO)
    assert tx.weights() is not None


def test_tdata_compute_weights_label():
    xs, yt = load_spirals()
    xl, yl = np.ones(len(xs)), np.ones(len(yt))
    xl[:50] = 0
    yl[:50] = 0
    tx, ty = TData(xs, labels=xl), TData(yt, labels=yl)
    tx.compute_weights(method=TR_WS_LABELS, other=ty)
    assert abs(tx.weights()[xl == 0].sum() - ty.weights()[yl == 0].sum()) < 1e-6
    assert abs(tx.weights()[xl == 1].sum() - ty.weights()[yl == 1].sum()) < 1e-6


def test_tdata_barycenter():
    xs, yt = load_spirals()
    t = TData(xs)
    b1 = t.get_barycenter()
    t.compute_weights(method=TR_WS_UNIFORM)
    b2 = t.get_barycenter()
    assert_array_almost_equal(b1, b2)


def test_transmorph_str_log_no_crash():
    t = Transmorph()
    str(t)
    t._log("foo")
    t._log("foo", end='bar')
    t._log("foo", header=False)


def test_tdata_str_log_no_crash():
    xs, _ = load_spirals()
    t = TData(xs)
    str(t)
    t._log("foo")
    t._log("foo", end='bar')
    t._log("foo", header=False)


def test_fit_empty_xs():
    # Transmorph raises an exception when fitted and xs is empty
    xs, yt = load_cell_cycle()
    tm = Transmorph()
    with pytest.raises(AssertionError):
        tm.fit(np.array([]), yt)


def test_fit_empty_yt():
    # Transmorph raises an exception when fitted and yt is empty
    xs, yt = load_cell_cycle()
    tm = Transmorph()
    with pytest.raises(AssertionError):
        tm.fit(xs, np.array([]))


def test_ot():
    # Optimal transport based dataset integration
    # Minimal setup
    # Spirals dataset (d=3)
    xs, yt = load_spirals()
    tm = Transmorph()
    xt = tm.fit_transform(xs, yt, jitter=False)
    assert xt.shape == xs.shape, \
        "Shape inconsistency: (%i, %i) != (%i, %i)" \
        % (*xt.shape, *xs.shape)

    
def test_ot_custom_xy_weights():
    # Optimal transport based dataset integration
    # Minimal setup
    # Spirals dataset (d=3)
    xs, yt = load_spirals()
    tm = Transmorph(weighting_strategy='uniform')
    xt1 = tm.fit_transform(xs, yt, jitter=False)
    xt2 = tm.fit_transform(xs, yt, jitter=False, xs_weights=np.ones(len(xs))/len(xs))
    xt3 = tm.fit_transform(xs, yt, jitter=False, yt_weights=np.ones(len(yt))/len(yt))
    assert_array_almost_equal(xt1, xt2)
    assert_array_almost_equal(xt1, xt3)


def test_ot_weighted():
    # Optimal transport based dataset integration
    # Weihgted setup
    # Spirals dataset (d=3)
    xs, yt = load_spirals()
    tm = Transmorph(
        weighting_strategy='woti',
    )
    xt = tm.fit_transform(xs, yt, jitter=False)
    assert xt.shape == xs.shape, \
        "Shape inconsistency: (%i, %i) != (%i, %i)" \
        % (*xt.shape, *xs.shape)


def test_ot_sinkhorn():
    # Optimal transport based dataset integration
    # Sinkhorn setup
    # Spirals dataset (d=3)
    xs, yt = load_spirals()
    tm = Transmorph(
        entropy=True
    )
    xt = tm.fit_transform(xs, yt, jitter=False)
    assert xt.shape == xs.shape, \
        "Shape inconsistency: (%i, %i) != (%i, %i)" \
        % (*xt.shape, *xs.shape)


def test_ot_unbalanced():
    # Optimal transport based dataset integration
    # Unbalanced setup
    # Spirals dataset (d=3)
    xs, yt = load_spirals()
    tm = Transmorph(
        unbalanced=True,
    )
    xt = tm.fit_transform(xs, yt, jitter=False)
    assert xt.shape == xs.shape, \
        "Shape inconsistency: (%i, %i) != (%i, %i)" \
        % (*xt.shape, *xs.shape)


def test_ot_custom_M():
    # Optimal transport based dataset integration
    # Minimal setup, custom cost matrix
    # Spirals dataset (d=3)
    xs, yt = load_spirals()
    tm = Transmorph(
        metric='sqeuclidean',
    )
    xt = tm.fit_transform(xs, yt, jitter=False)
    M = cdist(xs, yt, metric="sqeuclidean")
    xtM = tm.fit_transform(xs, yt, Mxy=M, jitter=False)
    assert_array_almost_equal(xt, xtM)


def test_ot_custom_M_wrong():
    # Optimal transport based dataset integration
    # Minimal setup, custom cost matrix
    # Should return an error
    xs, yt = load_spirals()
    tm = Transmorph(
        metric='sqeuclidean',
    )
    M = cdist(yt, xs, metric="sqeuclidean") # Wrong size
    with pytest.raises(AssertionError):
        xtM = tm.fit_transform(xs, yt, Mxy=M, jitter=False)


def test_ot_normalized():
    # Optimal transport based dataset integration
    # Normalization
    xs, yt = load_spirals()
    tm = Transmorph(
        method='ot',
    )
    tm.fit_transform(xs, yt)


def test_ot_pca_3d_2pc():
    # Optimal transport based dataset integration
    # Integration in PC space
    xs, yt = load_spirals()
    tm = Transmorph(
        method='ot',
        n_comps=2
    )
    tm.fit_transform(xs, yt)


def test_ot_pca_3d_2pc_normalized():
    # Optimal transport based dataset integration
    # Integration in PC space + normalization
    xs, yt = load_spirals()
    tm = Transmorph(
        method='ot',
        normalize=True,
        n_comps=2
    )
    tm.fit_transform(xs, yt)


def test_ot_pca_3d_3pc():
    # Optimal transport based dataset integration
    # Integration in PC space
    xs, yt = load_spirals()
    tm = Transmorph(
        method='ot',
        n_comps=3
    )
    tm.fit_transform(xs, yt)


def test_ot_pca_3d_4pc():
    # Optimal transport based dataset integration
    # Integration in PC space
    # Should return an error
    xs, yt = load_spirals()
    tm = Transmorph(
        method='ot',
        n_comps=4
    )
    with pytest.raises(AssertionError):
        tm.fit_transform(xs, yt)


def test_ot_subsample():
    xs, yt = load_spirals()
    tm = Transmorph(
        method='ot',
        n_hops=1
    )
    xt = tm.fit_transform(xs, yt)
    assert xt.shape == xs.shape, \
        "Shape inconsistency: (%i, %i) != (%i, %i)" \
        % (*xt.shape, *xs.shape)


def test_gromov():
    # Gromov-Wasserstein based dataset integration
    # Minimal setup
    # Spirals dataset (d=3)
    xs, yt = load_spirals()
    tm = Transmorph(
        method='gromov',
    )
    xt = tm.fit_transform(xs, yt, jitter=False)
    assert xt.shape == xs.shape, \
        "Shape inconsistency: (%i, %i) != (%i, %i)" \
        % (*xt.shape, *xs.shape)


def test_gromov_custom_Mx():
    # Gromov-Wasserstein based dataset integration
    # Minimal setup, custom cost matrix (source)
    # Spirals dataset (d=3)
    xs, yt = load_spirals()
    tm = Transmorph(
        method='gromov',
        metric='sqeuclidean'
    )
    xt = tm.fit_transform(xs, yt, jitter=False)
    M = cdist(xs, xs, metric='sqeuclidean')
    xtM = tm.fit_transform(xs, yt, Mx=M, jitter=False)
    assert_array_almost_equal(xt, xtM)


def test_gromov_custom_Mx_wrong():
    # Gromov-Wasserstein based dataset integration
    # Minimal setup, custom cost matrix (source)
    # Should return an error
    xs, yt = load_spirals()
    tm = Transmorph(
        method='gromov',
        metric='sqeuclidean'
    )
    M = cdist(yt, yt, metric='sqeuclidean')
    with pytest.raises(AssertionError):
        xtM = tm.fit_transform(xs, yt, Mx=M, jitter=False)


def test_gromov_custom_My():
    # Gromov-Wasserstein based dataset integration
    # Minimal setup, custom cost matrix (reference)
    # Spiral datasets (d=3)
    xs, yt = load_spirals()
    tm = Transmorph(
        method='gromov',
        metric='sqeuclidean'
    )
    xt = tm.fit_transform(xs, yt, jitter=False)
    M = cdist(yt, yt, metric='sqeuclidean')
    xtM = tm.fit_transform(xs, yt, My=M, jitter=False)
    assert_array_almost_equal(xt, xtM)


def test_gromov_custom_My_wrong():
    # Gromov-Wasserstein based dataset integration
    # Minimal setup, custom cost matrix (reference)
    # Should return an error
    xs, yt = load_spirals()
    tm = Transmorph(
        method='gromov',
        metric='sqeuclidean'
    )
    M = cdist(xs, xs, metric='sqeuclidean') # Wrong shape
    with pytest.raises(AssertionError):
        xtM = tm.fit_transform(xs, yt, My=M, jitter=False)


def test_gromov_geodesic():
    xs, yt = load_spirals()
    tm = Transmorph(
        method='gromov',
        geodesic=True
    )
    xt = tm.fit_transform(xs, yt)
    assert xt.shape == xs.shape, \
        "Shape inconsistency: (%i, %i) != (%i, %i)" \
        % (*xt.shape, *xs.shape)


def test_gromov_subsample():
    xs, yt = load_spirals()
    tm = Transmorph(
        method='gromov',
        n_hops=1
    )
    xt = tm.fit_transform(xs, yt)
    assert xt.shape == xs.shape, \
        "Shape inconsistency: (%i, %i) != (%i, %i)" \
        % (*xt.shape, *xs.shape)


def test_label_transfer():
    # Optimal transport based dataset integration
    # Minimal setup
    # Spirals dataset (d=3)
    xs, yt = load_spirals()
    tm = Transmorph()
    tm.fit(xs, yt)
    y_labels = np.ones(yt.shape[0])
    x_labels = tm.label_transfer(y_labels)
    assert x_labels.shape[0] == xs.shape[0], \
        "Shape inconsistency: %i != %i" \
        % (x_labels.shape[0], xs.shape[0])
