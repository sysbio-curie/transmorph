#!/usr/bin/env python3

import numpy as np

from transmorph.datasets import load_travaglini_10x
from transmorph.engine.transforming import CommonFeatures, Transformation


def test_transform_commonfeatures():
    # Tests standardize for all parameters sets
    databank = load_travaglini_10x()
    adatas = [adata for adata in databank.values()]
    # Computing target matrices
    common_features = adatas[0].var_names
    for adata in adatas[1:]:
        common_features = common_features.intersection(adata.var_names)
    targets = [adata[:, np.sort(common_features)].X.copy() for adata in adatas]
    # Building Transformation object
    transform = CommonFeatures()
    transform.retrieve_common_features(adatas, is_feature_space=True)
    # Source datasets
    datasets = [adata.X for adata in adatas]
    Transformation.assert_transform_equals(transform, adatas, datasets, targets)


if __name__ == "__main__":
    test_transform_commonfeatures()
