#!/usr/bin/env python3

import numpy as np

from scipy.sparse import csr_matrix
from transmorph.utils.matrix import sort_sparse_matrix, sparse_from_arrays


def test_sparse_sorting_and_recreating():
    # Data
    X = csr_matrix(
        np.array(
            #    0  1  2  3  4  5
            [
                [1, 0, 0, 3, 2, 0],
                [0, 2, 1, 0, 0, 4],
                [5, 4, 0, 0, 0, 1],
                [0, 0, 0, 1, 2, 3],
                [3, 2, 1, 0, 0, 0],
                [7, 8, 0, 0, 1, 0],
            ]
        )
    )
    X_nodata = csr_matrix(
        np.array(
            #    0  1  2  3  4  5
            [
                [1, 0, 0, 1, 1, 0],
                [0, 1, 1, 0, 0, 1],
                [1, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 1, 1],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 1, 0],
            ]
        )
    )
    target_indices = np.array(
        [[3, 4, 0], [5, 1, 2], [0, 1, 5], [5, 4, 3], [0, 1, 2], [1, 0, 4]]
    )
    target_data = np.array(
        [[3, 2, 1], [4, 2, 1], [5, 4, 1], [3, 2, 1], [3, 2, 1], [8, 7, 1]]
    )

    # Testing
    indices, data = sort_sparse_matrix(X, reverse=True)
    np.testing.assert_array_equal(indices, target_indices)
    np.testing.assert_array_equal(data, target_data)
    X_rebuilt = sparse_from_arrays(indices, data)
    np.testing.assert_array_equal(X.toarray(), X_rebuilt.toarray())
    X_rebuilt = sparse_from_arrays(indices)
    np.testing.assert_array_equal(X_nodata.toarray(), X_rebuilt.toarray())


if __name__ == "__main__":
    test_sparse_sorting_and_recreating()
