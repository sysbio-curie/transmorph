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
