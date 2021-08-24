#!/usr/bin/env python3

import numpy as np
import ot

from numba import njit
from scipy.spatial.distance import cdist
from sklearn.utils import check_array

from .transmorph import Transmorph
from .tdata import TData
from .density import kernel_H


def sigma_analysis(dataset,
                   layer='raw',
                   subsample=False,
                   sigmas: np.ndarray=None,
                   sigma_min: float=1e-10,
                   sigma_max: float=1e10,
                   nmarks: int=20,
                   log_scale: bool=True,
                   use_reference_if_transmorph: bool=False,
                   normalize_if_raw: bool=True,
                   return_sigmas: bool=False):
    """
    Returns the discretized mapping sigma -> KL(u_\\sigma, Uniform).
    The module uses

    argmax_\\sigma KL(u_\\sigma, Uniform)

    as a proxy for the optimal Gaussian kernel bandwidth.
    See more details in Fouché, bioRxiv 2021.

    Parameters:
    -----------
    dataset: Transmorph, TData or np.ndarray
        Dasaset to use for bandwidth selection. If a Transmorph
        object is passed, source TData is used by default. Reference
        TData can be used setting to True the parameter
        `use_reference_if_transmorph`.

    layer: str, default = 'raw'
        If dataset is Transmorph or TData, which layer to use,
        e.g. 'raw', 'pca'...

    subsample: bool, default = False
        If dataset is Transmorph or TData, use a subsample of the
        vertices. Warning: this changes the optimal value for sigma.

    sigmas: np.ndarray
        The set of sigmas to be tested. If None, a set of sigmas
        is selected automatically using the following parameters.

    sigma_min: float, default = 1e-10
        Lower bound for sigma search (included)

    sigma_max: float, default = 1e10
        Upper bound for sigma search (excluded)

    nmarks: int, default = 20
        Number of values to test.

    log_scale: bool, default = True
        Log-scaled grid.

    use_reference_if_transmorph: bool, default = False
        Use reference TData instead of source one, if dataset
        is a Transmorph.

    normalize_if_raw: bool, default = True
        Normalize columns if dataset is a np.ndarray.

    return_sigmas: bool, default = False
        Returns {sigmas}, {values} instead of {values}.
    """

    # Handling Transmorph, TData & np.ndarray
    if isinstance(dataset, Transmorph):
        assert dataset.fitted, \
            "Error: Transmorph is not fitted."
        if use_reference_if_transmorph:
            dataset = dataset.tdata_y
        else:
            dataset = dataset.tdata_x
    elif isinstance(dataset, np.ndarray):
        dataset = check_array(dataset, dtype=np.float32, order='C')
        dataset = TData(
            dataset,
            weights=None,
            labels=None,
            normalize=normalize_if_raw
        )
    else:
        assert isinstance(dataset, TData),\
            "Error: Unrecognized dataset type."
            
    # Computing an evenly spaced grid if none is provided
    if sigmas is None:
        if log_scale:
            sigma_min, sigma_max =\
                np.log(sigma_min), np.log(sigma_max)
        sigmas = np.arange(
            sigma_min,
            sigma_max,
            (sigma_max - sigma_min) / nmarks
        )
        if log_scale:
            sigmas = np.exp(sigmas)

    # Delegating to numba-accelerated function
    values = _sigma_analysis(
        dataset.distance(
            metric="euclidean",
            layer=layer,
            subsample=subsample,
            return_full_size=False
        ),
        sigmas
    )

    if return_sigmas:
        return sigmas, values

    return values
    

@njit
def _sigma_analysis(D, sigmas):
    values = []
    for sigma in sigmas:
        values.append(kernel_H(D, sigma))
    return values


def wasserstein_distance(tr: Transmorph,
                         x_integrated: np.ndarray = None,
                         use_labels: bool = False,
                         coefficient_labels: float = 1,
                         categorical_labels: bool = False,
                         xs_labels: np.ndarray = None,
                         yt_labels: np.ndarray = None,
                         layer: str = 'raw',
                         metric: str = 'sqeuclidean'):
    """
    Returns the total cost of transport matrix from a fitted
    Transmorph. 

    Parameters:
    -----------
    tr: Transmorph
        Transmorph, must be fitted

    x_integrated: np.ndarray, default = None
        Integrated dataset

    layer: str, default = 'raw'
        In 'raw', 'pca'. Representation to use, must have been
        precomputed in Transmorph.

    metric: str, default = 'sqeuclidean'
        Metric to use for cost matrix.
    """
    assert tr.fitted, \
        "Error: Transmorph not fitted."
    assert layer in ('raw', 'pca'), \
        "Layer %s not handled." % layer

    if x_integrated is None:
        x_integrated = tr.transform(jitter=False)

    yt = tr.tdata_y.get_layer(layer)
    xt = (
        x_integrated if layer == 'raw'
        else x_integrated @ tr.tdata_y.extras['pca'].T
    )

    M = cdist(xt, yt, metric=metric)

    if use_labels:
        if xs_labels is None:
            assert tr.tdata_x.labels is not None, \
                "Error: no labels in source dataset."
            xs_labels = tr.tdata_x.labels
        else:
            assert len(xs_labels) == len(tr.tdata_x), \
                "Error: Inconsistency between source dataset size and \
                labels size (%i != %i)" \
                % (len(xs_labels), len(tr.tdata_x))

        if yt_labels is None:
            assert tr.tdata_y.labels is not None, \
                "Error: no labels in reference dataset."
            yt_labels = tr.tdata_y.labels
        else:
            assert len(yt_labels) == len(tr.tdata_y), \
                "Error: Inconsistency between reference dataset size and \
                labels size (%i != %i)" \
                % (len(yt_labels), len(tr.tdata_y))

        assert coefficient_labels >= 0, \
            "Label coefficient must be positive, found %f" % coefficient_labels

        if categorical_labels:
            L = (xs_labels[:,None] != yt_labels)
        else:
            L = (xs_labels[:,None] - yt_labels)**2

        M += coefficient_labels * L

    M /= np.max(M)
    M = check_array(M, dtype=np.float64, order='C')

    return ot.lp.emd2(
        np.ones(len(xt))/len(xt),
        np.ones(len(yt))/len(yt),
        M,
        numItermax=1e6
    )
wd = wasserstein_distance # alias


def distance_label_continuous(tr: Transmorph,
                              x_integrated: np.ndarray = None,
                              xs_labels: np.ndarray = None,
                              yt_labels: np.ndarray = None,
                              layer: str = 'raw',
                              metric: str = 'sqeuclidean',
                              cost_per_point: bool = False):
    """
    Returns the total cost of transport matrix from a fitted
    Transmorph. 

    Parameters:
    -----------
    tr: Transmorph
        Transmorph, must be fitted

    x_integrated: np.ndarray, default = None
        Integrated dataset

    layer: str, default = 'raw'
        In 'raw', 'pca'. Representation to use, must have been
        precomputed in Transmorph.

    metric: str, default = 'sqeuclidean'
        Metric to use for cost matrix.
    """
    assert tr.fitted, \
        "Error: Transmorph not fitted."
    assert layer in ('raw', 'pca'), \
        "Layer %s not handled." % layer

    if xs_labels is None:
        assert tr.tdata_x.labels is not None, \
            "Error: no labels in source dataset."
        xs_labels = tr.tdata_x.labels
    else:
        assert len(xs_labels) == len(tr.tdata_x), \
            "Error: Inconsistency between source dataset size and \
            labels size (%i != %i)" \
            % (len(xs_labels), len(tr.tdata_x))

    if yt_labels is None:
        assert tr.tdata_y.labels is not None, \
            "Error: no labels in reference dataset."
        yt_labels = tr.tdata_y.labels
    else:
        assert len(yt_labels) == len(tr.tdata_y), \
            "Error: Inconsistency between reference dataset size and \
            labels size (%i != %i)" \
            % (len(yt_labels), len(tr.tdata_y))

    if x_integrated is None:
        x_integrated = tr.transform(jitter=False)

    yt = tr.tdata_y.get_layer(layer)
    xt = (
        x_integrated if layer == 'raw'
        else x_integrated @ tr.tdata_y.extras['pca'].T
    )

    diff = np.abs(
        yt_labels - xs_labels[:,None]
    )
    yt_matched = yt[np.argsort(diff, axis=1)[:,0]]
    distances = np.diag(cdist(xt, yt_matched, metric=metric))
    if cost_per_point:
        return distances
    return np.sum(distances)

