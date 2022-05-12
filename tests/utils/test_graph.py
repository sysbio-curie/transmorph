#!/usr/bin/env python3

import numpy as np

from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from transmorph.datasets import load_test_datasets_small
from transmorph.utils.graph import (
    cluster_anndatas,
    combine_matchings,
    get_nearest_vertex_from_set,
)
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
    cluster_anndatas(datasets, use_rep="repr", n_neighbors=3)


def test_get_nearest_vertex_from_set():
    G = np.array(
        [
            [1, -1, -1],
            [0, 2, -1],
            [1, 3, 5],
            [2, 4, 5],
            [3, 5, -1],
            [2, 3, 4],
            [7, -1, -1],
            [6, -1, -1],
        ]
    )
    D = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 3],
            [1, 1, 2],
            [1, 1, 0],
            [3, 2, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=np.float32,
    )
    vset = np.zeros((8,), dtype=bool)
    vset[0] = True
    vset[5] = True

    expected = np.array([0, 0, 0, 5, 5, 5, -1, -1])
    np.testing.assert_array_equal(expected, get_nearest_vertex_from_set(G, D, vset))


if __name__ == "__main__":
    test_get_nearest_vertex_from_set()
