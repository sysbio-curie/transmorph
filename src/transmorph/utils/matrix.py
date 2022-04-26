#!/usr/bin/env python3

# Helper functions to work with np.array and
# sparse matrices.

import numpy as np

from scipy.sparse import csr_matrix
from typing import List, Tuple, Optional


def center_matrix(X: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Centers an array to mean 0 along the given axis.
    """
    if axis == 0:
        return X - X.mean(axis=axis)
    else:
        return X - X.mean(axis=axis)[:, None]


def scale_matrix(X: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Scales an array to std 1 along the given axis.
    Ignores empty rows/cols.
    """
    normalizer = np.std(X, axis=axis)
    normalizer[normalizer == 0.0] = 1.0
    if axis == 0:
        return X / normalizer
    else:
        return X / normalizer[:, None]


def extract_chunks(X: np.ndarray, chunk_sizes: List[int]) -> List[np.ndarray]:
    """
    Slices a large matrix into a list of smaller chunks.

    Parameters
    ----------
    X: np.ndarray
        (n1 + ... + nk, d) array to slice

    chunk_sizes: np.ndarray
        Chunk sizes [n1, ..., nk]
    """
    assert sum(chunk_sizes) == X.shape[0], (
        "Inconsistent chunk sizes do not match X shape. "
        f"Expected {X.shape[0]}, found {sum(chunk_sizes)}."
    )
    offset = 0
    result = []
    for size in chunk_sizes:
        result.append(X[offset : offset + size])
        offset += size
    return result


def contains_duplicates(X: np.ndarray) -> bool:
    """
    Detects if X has duplicate rows.
    """
    return X.shape[0] != np.unique(X, axis=0).shape[0]


def perturbate(X: np.ndarray, std: float = 0.01) -> np.ndarray:
    """
    Perturbates X row samples by a small amount, useful to unstick
    points.
    """
    assert std >= 0.0, "Negative std is not allowed."
    if std == 0.0:
        return X
    return X + np.random.normal(loc=0.0, scale=std, size=X.shape)


def sort_sparse_matrix(
    X: csr_matrix, reverse: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (indices, values) so that rows of sparse matrix X are sorted.
    Each row of X must have the same number of values. Useful to sort
    a neighbors matrix for instance, to easily slice $k nearest
    neighbors.
    """
    coef = -1 if reverse else 1
    nsamples = X.shape[0]
    nlinks = (X[0] > 0).sum()
    indices = np.zeros((nsamples, nlinks), dtype=int)
    values = np.zeros((nsamples, nlinks), dtype=X.dtype)
    for i in range(X.shape[0]):
        row = X.getrow(i).tocoo()
        order = np.argsort(coef * row.data)
        indices[i] = row.col[order]
        values[i] = row.data[order]
    return indices, values


def sparse_from_arrays(
    np_indices: np.ndarray,
    np_data: Optional[np.ndarray] = None,
) -> csr_matrix:
    """
    Builds a sparse matrix from a pair of arrays.
    """
    rows, cols, data = [], [], []
    for i_row, col_indices in enumerate(np_indices):
        for col_index, j_col in enumerate(col_indices):
            rows.append(i_row)
            cols.append(j_col)
            if np_data is None:
                data.append(True)
            else:
                data.append(np_data[i_row, col_index])
    shape = (np_indices.shape[0], np_indices.shape[0])
    return csr_matrix((data, (rows, cols)), shape=shape)
