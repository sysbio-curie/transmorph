#!/usr/bin/env python3

import numpy as np

from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from transmorph.datasets import load_test_datasets_small
from transmorph.utils.graph import (
    cluster_anndatas,
    combine_matchings,
    distance_to_knn,
    generate_qtree,
    get_nearest_vertex_from_set,
    qtree_mutual_nearest_neighbors,
    raw_mutual_nearest_neighbors,
)
from transmorph.utils.anndata_manager import anndata_manager as adm

NTRIES, NDIMS = 50, 10


def test_distance_to_knn():
    # Tests distance to knn on random distance inputs
    for _ in range(NTRIES):
        nx = np.random.randint(low=100, high=500)
        ny = np.random.randint(low=100, high=500)
        D = np.random.random((nx, ny))
        k = np.random.randint(low=2, high=min(nx, ny))

        # axis 0 (v)
        np.testing.assert_array_equal(
            distance_to_knn(D, k, 0),
            np.sort(D, axis=0)[k - 1, :],
        )
        np.testing.assert_array_equal(
            distance_to_knn(D, 0, 0),
            np.zeros(ny, dtype=np.float32),
        )

        # axis 1 (>)
        np.testing.assert_array_equal(
            distance_to_knn(D, k, 1),
            np.sort(D, axis=1)[:, k - 1],
        )
        np.testing.assert_array_equal(
            distance_to_knn(D, 0, 1),
            np.zeros(nx, dtype=np.float32),
        )


def test_generate_qtree():
    # Generates qtrees from various random datasets
    for _ in range(NTRIES):
        nx = np.random.randint(low=100, high=500)
        X = np.random.random((nx, NDIMS))
        generate_qtree(X, "sqeuclidean", {})


def test_mutual_nearest_neighbors():
    # Tests MNN between various random datasets
    ntries = max(5, int(NTRIES / 10))
    for _ in range(ntries):
        nx = np.random.randint(low=100, high=500)
        X = np.random.random((nx, NDIMS))
        ny = np.random.randint(low=100, high=500)
        Y = np.random.random((ny, NDIMS))
        qtreex = generate_qtree(X, "sqeuclidean", {})
        qtreey = generate_qtree(Y, "sqeuclidean", {})
        mnn_approx = qtree_mutual_nearest_neighbors(X, Y, qtreex, qtreey, 10)
        mnn_exact = raw_mutual_nearest_neighbors(
            X,
            Y,
            metric="sqeuclidean",
            n_neighbors=10,
        )
        assert mnn_approx.shape == mnn_exact.shape == (nx, ny)
        nnbs = min(mnn_approx.count_nonzero(), mnn_exact.count_nonzero())
        mnn_diffs = mnn_approx - mnn_exact
        mnn_diffs.data = np.abs(mnn_diffs.data)
        assert mnn_diffs.sum() / nnbs < 0.1


def test_combine_matchings_smallcase():
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
    matchings[1, 0] = matchings[0, 1].T
    matchings[2, 0] = matchings[0, 2].T
    matchings[2, 1] = matchings[1, 2].T

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


def test_get_nearest_vertex_from_set_smallcase():
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
    test_combine_matchings_smallcase()
