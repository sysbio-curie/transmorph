#!/usr/bin/env python3

import numpy as np

from transmorph.datasets import load_test_datasets_small
from transmorph.engine.traits import UsesNeighbors
from transmorph.engine.subsampling import VertexCover
from transmorph.utils import AnnDataKeyIdentifiers

EXPECTED_RESULT = [
    (
        np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 0], dtype=bool),
        np.array([1, 1, 1, 1, 4, 1, 6, 6, 6, 6]),
    ),
    (
        np.array([0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=bool),
        np.array([1, 1, 1, 1, 4, 4, 4, 1, 4]),
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


if __name__ == "__main__":
    test_subsampling_vertexcover()
