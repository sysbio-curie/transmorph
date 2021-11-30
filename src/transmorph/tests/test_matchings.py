#!/usr/bin/env python3

import numpy as np
from scipy.sparse import csr_matrix

from numpy.testing import assert_array_almost_equal

from ..datasets import load_spirals
from ..matching import MatchingEMD


def test_MatchingEMD():
    """
    Test function for test related to matching.
    """
    xs, yt = load_spirals()
    matching = MatchingEMD()
    matching.fit(xs, reference=yt)
    assert matching.fitted
    assert matching.use_reference is True
    result_ref = matching.get_matching(0)
    assert type(result_ref) is csr_matrix
    matching.fit([xs, yt])
    assert matching.fitted
    assert matching.use_reference is False
    result_noref = matching.get_matching(0, 1)
    assert type(result_noref) is csr_matrix
    assert_array_almost_equal(result_ref.toarray(), result_noref.toarray())
    matching = MatchingEMD(metric="cosine")  # Test metric argument
    matching.fit(xs, reference=yt)
    assert matching.fitted
    matching = MatchingEMD(max_iter=int(1e7))  # Test max_iter argument
    matching.fit(xs, reference=yt)
    assert matching.fitted
    matching = MatchingEMD(use_sparse=False)  # Test sparsity argument
    matching.fit(xs, reference=yt)
    assert matching.fitted
    result = matching.get_matching(0)
    assert type(result) == np.ndarray
