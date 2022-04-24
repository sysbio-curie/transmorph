#!/usr/bin/env python3

import numpy as np

from transmorph.datasets import load_test_datasets_small
from transmorph.engine.transforming import Standardize, Transformation


# Small helper functions
def center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0)


def scale(X: np.ndarray) -> np.ndarray:
    normalizer = np.std(X, axis=0)
    normalizer[normalizer == 0.0] = 1.0
    return X / normalizer


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
    targets = [center(X) for X in datasets]
    transform = Standardize(center=True, scale=False)
    Transformation.assert_transform_equals(transform, datasets, targets)


def test_transform_standardize_scale():
    # Tests standardize for all parameters sets
    databank = load_test_datasets_small()
    datasets = [adata.X for adata in databank.values()]
    targets = [scale(X) for X in datasets]
    transform = Standardize(center=False, scale=True)
    Transformation.assert_transform_equals(transform, datasets, targets)


def test_transform_standardize_centerscale():
    # Tests standardize for all parameters sets
    databank = load_test_datasets_small()
    datasets = [adata.X for adata in databank.values()]
    targets = [scale(center(X)) for X in datasets]
    transform = Standardize(center=True, scale=True)
    Transformation.assert_transform_equals(transform, datasets, targets)


if __name__ == "__main__":
    test_transform_standardize_nochange()
    test_transform_standardize_center()
    test_transform_standardize_scale()
    test_transform_standardize_centerscale()
