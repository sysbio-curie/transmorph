#!/usr/bin/env python3

import numpy as np
from scipy.sparse import csr_matrix

from ..datasets import load_spirals
from ..matching import MatchingEMD


def test_MatchingEMD():
    """
    Test function for test related to matching.
    """
    xs, yt = load_spirals()
    matching = MatchingEMD()
    matching.match(xs, yt)
    assert matching.fitted
    result = matching.get(0, 1)
    assert type(result) == csr_matrix
    matching = MatchingEMD(metric="cosine")  # Test metric argument
    matching.match(xs, yt)
    assert matching.fitted
    matching = MatchingEMD(max_iter=int(1e7))  # Test max_iter argument
    matching.match(xs, yt)
    assert matching.fitted
    matching = MatchingEMD(use_sparse=False)  # Test sparsity argument
    matching.match(xs, yt)
    assert matching.fitted
    result = matching.get(0, 1)
    assert type(result) == np.ndarray
