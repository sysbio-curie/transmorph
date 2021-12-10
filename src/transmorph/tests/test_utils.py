#!/usr/bin/env python3

from ..datasets import load_spirals
from ..utils import nearest_neighbors


def test_nearest_neighbors():
    """
    Just executing functions for now. TODO: quality control
    """
    X, Y = load_spirals()
    nearest_neighbors(X)
    nearest_neighbors(X, use_nndescent=True)
    nearest_neighbors(X, Y)


def test_vertex_cover():
    pass


def test_pca():
    pass


def test_pca_multi():
    pass


def test_matching_divergence():
    pass
