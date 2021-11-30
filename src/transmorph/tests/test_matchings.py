#!/usr/bin/env python3

import numpy as np
from scipy.sparse import csr_matrix

from numpy.testing import assert_array_almost_equal

from ..datasets import load_spirals
from ..matching import MatchingEMD


def _generic_matching_test(matching_class):
    """
    Verify a given matching verifies abstraction requirements.
    """
    xs, yt = load_spirals()
    matching = matching_class()
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
    matching = matching_class(use_sparse=False)  # Test sparsity argument
    matching.fit(xs, reference=yt)
    assert matching.fitted
    result = matching.get_matching(0)
    assert type(result) == np.ndarray


def test_MatchingEMD():
    """
    Test function for test related to matching.
    """
    _generic_matching_test(MatchingEMD)
