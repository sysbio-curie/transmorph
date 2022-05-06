#!/usr/bin/env python3

from typing import Callable

from ...stats.matching import edge_accuracy


def matching_edge_accuracy(label: str) -> Callable:
    """
    This evaluator measures the frequency of edges in a matching
    between two datasets that bind samples from the same class.

    Parameters
    ----------
    label: str
        .obs column entry containing labels.
    """
    return lambda a1, a2, T: edge_accuracy(a1, a2, T, label)
