#!/usr/bin/env python3

import numpy as np

from transmorph.datasets import load_travaglini_10x
from transmorph.engine.traits import UsesCommonFeatures


travaglini = list(load_travaglini_10x().values())


def test_usescommonfeatures_total():
    # Tests UsesCommonFeatures for mode = "total"
    comfeat = UsesCommonFeatures(mode="total")
    comfeat.retrieve_common_features(travaglini, True)
    comfeat.assert_common_features(travaglini)


def test_usescommonfeatures_pairwise():
    # Tests UsesCommonFeatures for mode = "pairwise"
    comfeat = UsesCommonFeatures(mode="pairwise")
    comfeat.retrieve_common_features(travaglini, True)
    comfeat.assert_common_features(travaglini)


def test_usescommonfeatures_array_slice_total():
    # Tests if sliced arrays correspond to sliced features.
    comfeat = UsesCommonFeatures(mode="total")
    comfeat.retrieve_common_features(travaglini, True)
    for i, adata in enumerate(travaglini):
        X_sliced = comfeat.slice_features(adata.X, i)
        np.testing.assert_array_equal(
            adata.X[:, comfeat.total_feature_slices[i]],
            X_sliced,
        )


def test_usescommonfeatures_array_slice_pairwise():
    # Tests if sliced arrays correspond to sliced features.
    comfeat = UsesCommonFeatures(mode="pairwise")
    comfeat.retrieve_common_features(travaglini, True)
    for i, adata_i in enumerate(travaglini):
        for j, adata_j in enumerate(travaglini):
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
