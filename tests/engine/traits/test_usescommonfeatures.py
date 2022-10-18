#!/usr/bin/env python3

import numpy as np

from transmorph.datasets import load_test_datasets_random
from transmorph.engine.traits import UsesCommonFeatures


datasets = list(load_test_datasets_random().values())


def test_usescommonfeatures_total():
    # Tests UsesCommonFeatures for mode = "total"
    comfeat = UsesCommonFeatures(mode="total")
    comfeat.retrieve_common_features(datasets, True)
    comfeat.assert_common_features(datasets)


def test_usescommonfeatures_pairwise():
    # Tests UsesCommonFeatures for mode = "pairwise"
    comfeat = UsesCommonFeatures(mode="pairwise")
    comfeat.retrieve_common_features(datasets, True)
    comfeat.assert_common_features(datasets)


def test_usescommonfeatures_array_slice_total():
    # Tests if sliced arrays correspond to sliced features.
    comfeat = UsesCommonFeatures(mode="total")
    comfeat.retrieve_common_features(datasets, True)
    for i, adata in enumerate(datasets):
        X_sliced = comfeat.slice_features(adata.X, i)
        np.testing.assert_array_equal(
            adata.X[:, comfeat.total_feature_slices[i]],
            X_sliced,
        )


def test_usescommonfeatures_array_slice_pairwise():
    # Tests if sliced arrays correspond to sliced features.
    comfeat = UsesCommonFeatures(mode="pairwise")
    comfeat.retrieve_common_features(datasets, True)
    for i, adata_i in enumerate(datasets):
        for j, adata_j in enumerate(datasets):
            Xi_sliced, Xj_sliced = comfeat.slice_features(adata_i.X, i, adata_j.X, j)
            slice_i, slice_j = comfeat.pairwise_feature_slices[i, j]
            np.testing.assert_array_equal(
                adata_i.X[:, slice_i],
                Xi_sliced,
            )
            np.testing.assert_array_equal(
                adata_j.X[:, slice_j],
                Xj_sliced,
            )


if __name__ == "__main__":
    test_usescommonfeatures_array_slice_pairwise()
