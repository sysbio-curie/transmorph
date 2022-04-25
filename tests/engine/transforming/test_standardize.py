#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small
from transmorph.engine.transforming import Standardize, Transformation
from transmorph.utils.matrix import center_matrix, scale_matrix


def test_transform_standardize_nochange():
    # Tests standardize for all parameters sets
    databank = load_test_datasets_small()
    datasets = [adata.X for adata in databank.values()]
    targets = [X.copy() for X in datasets]
    transform = Standardize(center=False, scale=False)
    Transformation.assert_transform_equals(transform, datasets, targets)


def test_transform_standardize_center():
    # Tests standardize for all parameters sets
    databank = load_test_datasets_small()
    datasets = [adata.X for adata in databank.values()]
    targets = [center_matrix(X) for X in datasets]
    transform = Standardize(center=True, scale=False)
    Transformation.assert_transform_equals(transform, datasets, targets)


def test_transform_standardize_scale():
    # Tests standardize for all parameters sets
    databank = load_test_datasets_small()
    datasets = [adata.X for adata in databank.values()]
    targets = [scale_matrix(X) for X in datasets]
    transform = Standardize(center=False, scale=True)
    Transformation.assert_transform_equals(transform, datasets, targets)


def test_transform_standardize_centerscale():
    # Tests standardize for all parameters sets
    databank = load_test_datasets_small()
    datasets = [adata.X for adata in databank.values()]
    targets = [scale_matrix(center_matrix(X)) for X in datasets]
    transform = Standardize(center=True, scale=True)
    Transformation.assert_transform_equals(transform, datasets, targets)


if __name__ == "__main__":
    test_transform_standardize_nochange()
    test_transform_standardize_center()
    test_transform_standardize_scale()
    test_transform_standardize_centerscale()
