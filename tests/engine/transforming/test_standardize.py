#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small
from transmorph.engine.transforming import Standardize, Transformation
from transmorph.utils.matrix import center_matrix, scale_matrix


def test_transform_standardize_nochange():
    # Tests standardize for all parameters sets
    datasets = list(load_test_datasets_small().values())
    embeddings = [adata.X for adata in datasets]
    targets = [X.copy() for X in embeddings]
    transform = Standardize(center=False, scale=False)
    Transformation.assert_transform_equals(
        transform,
        datasets,
        embeddings,
        targets,
    )


def test_transform_standardize_center():
    # Tests standardize for all parameters sets
    datasets = list(load_test_datasets_small().values())
    embeddings = [adata.X for adata in datasets]
    targets = [X.copy() for X in embeddings]
    transform = Standardize(center=False, scale=False)
    Transformation.assert_transform_equals(
        transform,
        datasets,
        embeddings,
        targets,
    )


def test_transform_standardize_scale():
    # Tests standardize for all parameters sets
    datasets = list(load_test_datasets_small().values())
    embeddings = [scale_matrix(adata.X) for adata in datasets]
    targets = [center_matrix(X.copy()) for X in embeddings]
    transform = Standardize(center=True, scale=True)
    Transformation.assert_transform_equals(
        transform,
        datasets,
        embeddings,
        targets,
    )


def test_transform_standardize_centerscale():
    # Tests standardize for all parameters sets
    datasets = list(load_test_datasets_small().values())
    embeddings = [adata.X for adata in datasets]
    targets = [scale_matrix(center_matrix(X)) for X in embeddings]
    transform = Standardize(center=True, scale=True)
    Transformation.assert_transform_equals(
        transform,
        datasets,
        embeddings,
        targets,
    )


if __name__ == "__main__":
    test_transform_standardize_nochange()
    test_transform_standardize_center()
    test_transform_standardize_scale()
    test_transform_standardize_centerscale()
