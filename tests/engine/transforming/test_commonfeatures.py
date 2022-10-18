#!/usr/bin/env python3

import numpy as np

from transmorph.datasets import load_test_datasets_random
from transmorph.engine.transforming import CommonFeatures, Transformation


def test_transform_commonfeatures():
    # Tests standardize for all parameters sets
    datasets = list(load_test_datasets_random().values())
    # Computing target matrices
    common_features = datasets[0].var_names
    for adata in datasets[1:]:
        common_features = common_features.intersection(adata.var_names)
    targets = [adata[:, np.sort(common_features)].X.copy() for adata in datasets]
    # Building Transformation object
    transform = CommonFeatures()
    transform.retrieve_common_features(datasets, is_feature_space=True)
    # Source datasets
    embeddings = [adata.X for adata in datasets]
    Transformation.assert_transform_equals(transform, datasets, embeddings, targets)


if __name__ == "__main__":
    test_transform_commonfeatures()
