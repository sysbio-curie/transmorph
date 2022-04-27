#!/usr/bin/env python3

import numpy as np
from numpy.testing._private.utils import assert_array_equal

from scipy.sparse import csr_matrix
from transmorph.utils.graph import combine_matchings


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


if __name__ == "__main__":
    test_combine_matchings()
