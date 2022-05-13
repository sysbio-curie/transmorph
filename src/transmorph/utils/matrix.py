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


def scale_matrix(X: np.ndarray, axis: int = 0, std_mode: bool = True) -> np.ndarray:
    """
    Scales an array to std 1 along the given axis.
    Ignores empty rows/cols.
    """
    if std_mode:  # std mode
        normalizer = np.std(X, axis=axis)
    else:  # sum mode
        normalizer = np.sum(X, axis=axis)
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


def perturbate(X: np.ndarray, std: float = 0.01, inplace: bool = False) -> np.ndarray:
    """
    Perturbates X row samples by a small amount, useful to unstick
    points.
    """
    assert std >= 0.0, "Negative std is not allowed."
    if std == 0.0:
        return X
    return X * (1.0 + np.random.normal(loc=0.0, scale=std, size=X.shape))


def sort_sparse_matrix(
    X: csr_matrix,
    reverse: bool = False,
    fill_empty: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (indices, values) so that rows of sparse matrix X are sorted.
    Useful to sort a neighbors matrix for instance, to easily slice $k
    nearest neighbors. If fill_empty = True, different number of elements
    between rows is allowed. Gaps are filled with np.inf.
    """
    coef = -1.0 if reverse else 1.0
    nsamples = X.shape[0]
    if fill_empty:
        nlinks = np.max((X > 0).sum(axis=1))
    else:
        nlinks = (X[0] > 0).sum()
        assert np.all((X != 0).sum(axis=1) == nlinks), (
            "Inconsistent number of elements in rows detected. If "
            "this is expected, please explicitly set fill_empty to True."
        )
    indices = np.zeros((nsamples, nlinks), dtype=int) - 1  # No value -> -1
    values = np.zeros((nsamples, nlinks), dtype=np.float32) + np.inf  # No value -> inf
    for i in range(X.shape[0]):
        row = X.getrow(i).tocoo()
        order = np.argsort(coef * row.data)
        k = row.data.shape[0]
        indices[i, :k] = row.col[order]
        values[i, :k] = row.data[order]
    return indices, values


def sparse_from_arrays(
    np_indices: np.ndarray,
    np_data: Optional[np.ndarray] = None,
    n_cols: Optional[int] = None,
) -> csr_matrix:
    """
    Builds a sparse matrix from a pair of arrays.
    If np_data is None, nonzero values will be set to True.
    If n_cols is None, matrix is assumed to be square.
    """
    rows, cols, data = [], [], []
    for i_row, col_indices in enumerate(np_indices):
        for col_index, j_col in enumerate(col_indices):
            if j_col == -1:
                continue
            rows.append(i_row)
            cols.append(j_col)
            if np_data is None:
                data.append(1.0)
            else:
                data.append(np_data[i_row, col_index])
    if n_cols is None:
        n_cols = np_indices.shape[0]
    shape = (np_indices.shape[0], n_cols)
    return csr_matrix((data, (rows, cols)), shape=shape)


def pooling(X: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Replaces each sample by the average of its neighbors. This
    is a useful to smooth data manifolds, and reduce the impact
    of outliers or artifacts.
    """
    return X[indices].mean(axis=1)


def guess_is_discrete(X: np.ndarray) -> bool:
    """
    Returns True if X content is compatible with a discrete
    label.
    """
    from .._settings import settings

    return np.unique(X).shape[0] < settings.is_discrete_unique_thr
