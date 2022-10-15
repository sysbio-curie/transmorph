#!/usr/bin/env python3

import numpy as np

from sklearn import decomposition as skd
from transmorph import settings
from transmorph.datasets import load_travaglini_10x
from transmorph.engine import transforming as trf
from transmorph.utils.matrix import extract_chunks

N_PCS = 20


def test_transform_pca():
    # Tests standardize for all parameters sets
    databank = load_travaglini_10x()
    adatas = [adata for adata in databank.values()]
    # PCA uses common features
    common_features = adatas[0].var_names
    for adata in adatas[1:]:
        common_features = common_features.intersection(adata.var_names)
    to_reduce = [adata[:, np.sort(common_features)].X for adata in adatas]
    targets = extract_chunks(
        skd.PCA(
            n_components=N_PCS, random_state=settings.global_random_seed
        ).fit_transform(np.concatenate(to_reduce, axis=0)),
        [adata.n_obs for adata in adatas],
    )
    transform = trf.PCA(n_components=N_PCS)
    transform.retrieve_common_features(adatas, is_feature_space=True)
    datasets = [adata.X for adata in adatas]
    # Random PCA solver means we decrease the # of decimals required.
    trf.Transformation.assert_transform_equals(
        transform,
        adatas,
        datasets,
        targets,
        decimal=1,
    )


if __name__ == "__main__":
    test_transform_pca()
