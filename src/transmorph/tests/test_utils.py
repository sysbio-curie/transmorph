#!/usr/bin/env python3

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest as pt

from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

from ..datasets import load_spirals
from ..utils import nearest_neighbors, pca, pca_multi, vertex_cover


X, Y = load_spirals()
n, m = X.shape[0], Y.shape[0]


def test_nearest_neighbors():
    # Nearest neighbors
    n_neighbors = 10
    nn1 = nearest_neighbors(X, n_neighbors=n_neighbors, symmetrize=False)
    assert type(nn1) is csr_matrix
    assert nn1.shape == (n, n)
    assert nn1.sum(axis=1).max() == nn1.sum(axis=1).min() == n_neighbors
    nn2 = nearest_neighbors(
        X, use_nndescent=True, n_neighbors=n_neighbors, symmetrize=False
    )
    assert type(nn2) is csr_matrix
    assert nn2.shape == (n, n)
    assert nn2.sum(axis=1).max() == nn2.sum(axis=1).min() == n_neighbors

    # Symmetrized nearest neighbors
    nn3 = nearest_neighbors(X, n_neighbors=n_neighbors, symmetrize=True)
    assert type(nn3) is csr_matrix
    assert nn3.shape == (n, n)
    assert abs(nn3 - nn3.T).max() == 0

    # Mutual nearest neighbors
    with pt.raises(AssertionError):
        mnn = nearest_neighbors(X, Y=Y, use_nndescent=True)
    mnn = nearest_neighbors(X, Y=Y)
    assert mnn.shape == (n, m)


def test_vertex_cover():
    nn = nearest_neighbors(X, n_neighbors=10)
    with pt.raises(AssertionError):
        vertex_cover(nn, hops=-1)
    vc0, mp0 = vertex_cover(nn, hops=0)
    vc1, mp1 = vertex_cover(nn, hops=1)
    vc2, mp2 = vertex_cover(nn, hops=2)

    # Testing output shape
    assert vc0.shape[0] == vc1.shape[0] == vc2.shape[0] == n
    assert mp0.shape[0] == mp1.shape[0] == mp2.shape[0] == n
    assert_array_almost_equal(vc0, np.ones(n))
    assert_array_almost_equal(mp0, np.arange(n))

    # Testing vertex cover quality
    assert vc0.sum() > vc1.sum() > vc2.sum()
    for i in range(n):
        assert vc0[mp0[i]] == 1
        assert vc1[mp1[i]] == 1
        assert vc2[mp2[i]] == 1


def test_pca():
    with pt.raises(AssertionError):
        pca(X, n_components=0)
    pca_sklearn = PCA(n_components=2).fit(X).transform(X)
    pca_transmorph = pca(X, n_components=2)
    _, pc_object = pca(X, n_components=2, return_transformation=True)
    assert_array_almost_equal(pca_sklearn, pca_transmorph)
    assert_array_almost_equal(pca_sklearn, pc_object.transform(X))


def test_pca_multi():
    with pt.raises(AssertionError):
        pca_multi([X, Y], n_components=0)
    with pt.raises(AssertionError):
        pca_multi([])
    with pt.raises(NotImplementedError):
        pca_multi([X, Y], strategy="whatever")
    with pt.raises(AssertionError):
        pca_multi([X, Y], strategy="independent", return_transformation=True)
    pca_multi(
        [X, Y], n_components=2, strategy="concatenate", return_transformation=True
    )
    pca_multi([X, Y], n_components=2, strategy="reference", return_transformation=True)
    pca_multi([X, Y], n_components=2, strategy="composite", return_transformation=True)
    pca_multi([X, Y], n_components=2, strategy="independent")


def test_earth_movers_distance():
    pass


def test_matching_divergence():
    pass


def test_neighborhood_preservation():
    pass
