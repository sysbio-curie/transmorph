#!/usr/bin/env python3

import numpy as np

from transmorph.utils.matrix import pooling


def test_transformation_pooling():
    X = np.array([[1, 0, 1], [3, 2, 1], [1, 2, 4], [5, 6, 1], [0, 0, 1]])
    indices = np.array([[1, 2], [3, 1], [1, 4], [0, 1], [2, 3]])
    target = np.array([[2, 2, 2.5], [4, 4, 1], [1.5, 1, 1], [2, 1, 1], [3, 4, 2.5]])
    np.testing.assert_array_equal(pooling(X, indices), target)


if __name__ == "__main__":
    test_transformation_pooling()
