#!/usr/bin/env python3

import numpy as np

from typing import List


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
