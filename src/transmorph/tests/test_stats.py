#!/usr/bin/env python3

import pytest
import numpy as np

from sklearn.utils._testing import assert_array_almost_equal

from ..constants import *
from ..datasets import load_cell_cycle
from ..datasets import load_spirals
from ..datasets import load_spirals_labels
from ..stats import distance_label_continuous
from ..stats import sigma_analysis
from ..stats import wasserstein_distance, wd
from ..tdata import TData
from ..transmorph import Transmorph

# Test testing metrics
# TODO: Test against reference values

def test_sigma_analysis():
    xs, yt = load_spirals()
    t = Transmorph(normalize=True)
    t.fit(xs, yt)
    values_x_transmorph = sigma_analysis(t)
    values_x_tdata = sigma_analysis(t.tdata_x)
    values_x_ndarray = sigma_analysis(xs, normalize_if_raw=True)
    
    values_y_transmorph = sigma_analysis(t, use_reference_if_transmorph=True)
    values_y_tdata = sigma_analysis(t.tdata_y)
    values_y_ndarray = sigma_analysis(yt, normalize_if_raw=True)

    assert_array_almost_equal(values_x_ndarray, values_x_tdata)
    assert_array_almost_equal(values_x_ndarray, values_x_transmorph)
    assert_array_almost_equal(values_y_ndarray, values_y_tdata)
    assert_array_almost_equal(values_y_ndarray, values_y_transmorph)


def test_wasserstein_distance():
    xs, yt = load_spirals()
    t = Transmorph(normalize=True)
    t.fit(xs, yt)
    xt = t.transform(jitter=False)
    wdistance = wasserstein_distance(t)
    wdistance2 = wasserstein_distance(t, xt)
    wdistance3 = wd(t)

    assert np.array_equal(wdistance, wdistance2)
    assert np.array_equal(wdistance, wdistance3)


def test_distance_label_continuous():
    xs, yt = load_spirals()
    xl, yl = load_spirals_labels()
    t = Transmorph(normalize=True)
    t.fit(xs, yt)
    xt = t.transform(jitter=False)
    ldistance = distance_label_continuous(
        t,
        xs_labels=xl,
        yt_labels=yl,
        cost_per_point=False
    )
    ldistance2 = distance_label_continuous(
        t,
        x_integrated=xt,
        xs_labels=xl,
        yt_labels=yl,
        cost_per_point=False
    )
    assert ldistance == ldistance2

    ldistance = distance_label_continuous(
        t,
        xs_labels=xl,
        yt_labels=yl,
        cost_per_point=True
    )
    ldistance2 = distance_label_continuous(
        t,
        x_integrated=xt,
        xs_labels=xl,
        yt_labels=yl,
        cost_per_point=True
    )
    assert np.array_equal(ldistance, ldistance2)
