#!/usr/bin/env python3

import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix

from numpy.testing import assert_array_almost_equal

from ..datasets.anndata import preprocess_anndata
from ..datasets import load_spirals
from ..matching import (
    MatchingEMD,
    MatchingSinkhorn,
    MatchingGW,
    MatchingGWEntropic,
    MatchingMNN,
    MatchingPartialOT,
    MatchingFusedGW,
)


def _generic_matching_test(matching_class):
    """
    Verify a given matching verifies abstraction requirements.
    """
    xs, yt = load_spirals()
    adata1 = preprocess_anndata(sc.AnnData(xs), {"metric": "euclidean"})
    adata2 = preprocess_anndata(sc.AnnData(yt), {"metric": "euclidean"})
    matching = matching_class()
    matching.fit([adata1], reference=adata2)
    assert matching.fitted
    assert matching.use_reference is True
    result_ref = matching.get_matching(0)
    assert type(result_ref) is csr_matrix
    print("matching with reference ok.")
    matching.fit([adata1, adata2])
    assert matching.fitted
    assert matching.use_reference is False
    result_noref = matching.get_matching(0, 1)
    assert type(result_noref) is csr_matrix
    assert_array_almost_equal(result_ref.toarray(), result_noref.toarray())
    print("matching with list ok.")
    matching = matching_class(use_sparse=False)  # Test sparsity argument
    matching.fit([adata1], reference=adata2)
    assert matching.fitted
    result = matching.get_matching(0)
    assert type(result) == np.ndarray
    print("matching with sparsity ok.")


def test_MatchingEMD():
    """
    Earth Mover's Distance-based matching
    """
    _generic_matching_test(MatchingEMD)


def test_MatchingSinkhorn():
    """
    Sinkhorn-Knopp-based matching
    """
    _generic_matching_test(MatchingSinkhorn)


def test_MatchingGW():
    """
    Gromov-Wasserstein-based matching
    """
    _generic_matching_test(MatchingGW)


def test_MatchingGWEntropic():
    """
    Entropic Gromov-Wasserstein-based matching
    """
    _generic_matching_test(MatchingGWEntropic)


def test_MatchingMNN():
    """
    Mutual Nearest Neighbors-based matching
    """
    _generic_matching_test(MatchingMNN)


def test_MatchingFusedGW():
    """
    Fused Gromov-Wasserstein-based matching
    """
    _generic_matching_test(MatchingFusedGW)


def test_MatchingPartialOT():
    """
    Partial OT-based matching
    """
    _generic_matching_test(MatchingPartialOT)
