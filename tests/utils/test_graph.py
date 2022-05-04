#!/usr/bin/env python3

import numpy as np

from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from transmorph.datasets import load_test_datasets_small
from transmorph.utils.graph import cluster_anndatas, combine_matchings
from transmorph.utils.anndata_manager import anndata_manager as adm


def test_combine_matchings():
    matchings = {
        (0, 1): np.array(
            [
                [1, 0, 0],
                [0, 1, 1],
            ]
        ),
        (0, 2): np.array(
            [
                [0, 1, 1, 1],
                [1, 0, 1, 0],
            ]
        ),
        (1, 2): np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
            ]
        ),
    }
    inner_graphs = [
        np.array(
            [
                [0, 1],
                [1, 0],
            ]
        ),
        np.array(
            [
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
            ]
        ),
        np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 1],
                [1, 0, 1, 0],
            ]
        ),
    ]
    target = np.array(
        [
            [0, 1, 1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 1, 1, 0, 1, 0],
            [1, 0, 0, 1, 1, 1, 0, 0, 1],
            [0, 1, 1, 0, 1, 0, 1, 1, 0],
            [0, 1, 1, 1, 0, 1, 0, 0, 1],
            [0, 1, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 1, 0, 1, 0],
            [1, 1, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 0, 1, 0],
        ]
    )
    for key in matchings:
        matchings[key] = csr_matrix(matchings[key])
    for i in range(len(inner_graphs)):
        inner_graphs[i] = csr_matrix(inner_graphs[i])
    T = combine_matchings(matchings, knn_graphs=inner_graphs)
    test = T.toarray().astype(int)
    assert_array_equal(test, target)


def test_cluster_anndatas():
    # Tests clustering on a small dataset
    datasets = list(load_test_datasets_small().values())
    for adata in datasets:
        adm.set_value(
            adata,
            key="repr",
            field="obsm",
            value=adata.X,
            persist="pipeline",
        )
    cluster_anndatas(datasets, use_rep="tr_repr", n_neighbors=3)


if __name__ == "__main__":
    test_cluster_anndatas()
