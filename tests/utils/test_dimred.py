#!/usr/bin/env python3

import numpy as np

from transmorph.utils.dimred import pca_projector, pca, ica, umap

NTRIES = 20


def test_pca_projector():
    for _ in range(NTRIES):
        X = np.random.random(size=(1000, 50))
        pca_obj = pca_projector(X, n_components=30)
        assert pca_obj.components_.shape == (30, X.shape[1])


def test_pca():
    for _ in range(NTRIES):
        X = np.random.random(size=(1000, 50))
        X_pca = pca(X, n_components=30)
        assert X_pca.shape == (X.shape[0], 30)


def test_ica():
    for _ in range(NTRIES):
        X = np.random.random(size=(1000, 50))
        X_ica = ica(X, n_components=30)
        assert X_ica.shape == (X.shape[0], 30)


def test_umap():
    for _ in range(NTRIES):
        X = np.random.random(size=(1000, 50))
        X_umap = umap(X, embedding_dimension=2)
        assert X_umap.shape == (X.shape[0], 2)


if __name__ == "__main__":
    test_ica()
