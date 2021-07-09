#!/usr/bin/env python3

import pytest
import numpy as np

from scipy.spatial.distance import cdist
from sklearn.utils._testing import assert_array_equal

from ..datasets import load_cell_cycle, load_spirals
from ..transmorph import Transmorph

# For now, tests will not challenge the return value
# They will rather test for code inconsistencies
# TODO: Think of testing metrics

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
    assert_array_equal(xt, xtM)


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
    assert_array_equal(xt, xtM)


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
    assert_array_equal(xt, xtM)


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
