#!/usr/bin/env python3

import numpy as np

from scipy.sparse import csr_matrix

from transmorph.datasets import load_test_datasets_small
from transmorph.engine.traits import IsSubsamplable, UsesNeighbors
from transmorph.engine.subsampling import VertexCover
from transmorph.utils import AnnDataKeyIdentifiers

EXPECTED_RESULT = [
    (
        np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 0], dtype=bool),
        np.array([1, 1, 1, 1, 4, 1, 6, 6, 6, 6]),
    ),
    (
        np.array([0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=bool),
        np.array([1, 1, 1, 1, 4, 4, 1, 4, 4]),
    ),
]


def test_subsampling_vertexcover():
    # Tests matching quality of partial OT on small controlled dataset
    datasets = list(load_test_datasets_small().values())
    UsesNeighbors.compute_neighbors_graphs(
        datasets=datasets,
        representation_key=AnnDataKeyIdentifiers.BaseRepresentation,
    )
    subsampling = VertexCover(n_neighbors=4)
    result = subsampling.subsample([adata.X for adata in datasets])
    for vc_true, vc_test in zip(EXPECTED_RESULT, result):
        a_true, r_true = vc_true
        a_test, r_test = vc_test
        np.testing.assert_array_equal(a_true, a_test)
        np.testing.assert_array_equal(r_true, r_test)


def test_subsample_unsubsample():
    # Tests unsubsample o subsample = identity
    datasets = list(load_test_datasets_small().values())
    UsesNeighbors.compute_neighbors_graphs(
        datasets=datasets, representation_key=AnnDataKeyIdentifiers.BaseRepresentation
    )
    subsampling = VertexCover(n_neighbors=2)
    issub = IsSubsamplable(subsampling=subsampling)
    issub.compute_subsampling(datasets, [adata.X for adata in datasets], True)
    # ndarray
    T = np.random.random((datasets[0].n_obs, datasets[1].n_obs))
    T_after = issub.unsubsample_matrix(issub.subsample_matrix(T, 0, 1, False), 0, 1)
    assert isinstance(T_after, np.ndarray)
    np.testing.assert_array_equal(
        issub.subsample_matrix(T, 0, 1),
        issub.subsample_matrix(T_after, 0, 1),
    )
    # csr
    T = csr_matrix(np.random.random((datasets[0].n_obs, datasets[1].n_obs)))
    T_after = issub.unsubsample_matrix(issub.subsample_matrix(T, 0, 1, False), 0, 1)
    assert isinstance(T_after, csr_matrix)
    np.testing.assert_array_equal(
        issub.subsample_matrix(T.toarray(), 0, 1),
        issub.subsample_matrix(T_after.toarray(), 0, 1),
    )


if __name__ == "__main__":
    test_subsampling_vertexcover()
    test_subsample_unsubsample()
