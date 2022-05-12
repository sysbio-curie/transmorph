#!/usr/bin/env python3

import numpy as np

from anndata import AnnData
from typing import Callable, List

from ...engine.layers import LayerMatching
from ...stats.matching import edge_accuracy, edge_penalty


def evaluate_matching_layer(
    layer: LayerMatching,
    datasets: List[AnnData],
    evaluator: Callable,
) -> np.ndarray:
    """
    layer: LayerMatching
        LayerMatching which has been fit, containing matchings to
        evaluate.

    datasets: List[AnnData]
        List of AnnData objects that have been passed to Model.fit().

    evaluator: Callable
        Evaluation metric f : AnnData, AnnData, csr_matrix -> float
        to use.
    """
    ndatasets = len(datasets)
    results = np.zeros((ndatasets, ndatasets), dtype=np.float32)
    matchings = layer.get_matchings()
    for i in range(ndatasets):
        for j in range(i + 1, ndatasets):
            results[i, j] = results[j, i] = evaluator(
                datasets[i], datasets[j], matchings[i, j]
            )
    return results


def matching_edge_accuracy_discrete(label: str) -> Callable:
    """
    This evaluator measures the frequency of edges in a matching
    between two datasets that bind samples from the same class.

    Parameters
    ----------
    label: str
        .obs column entry containing labels.
    """
    return lambda a1, a2, T: 1.0 if a1 is a2 else edge_accuracy(a1, a2, T, label)


def matching_edge_penalty_continuous(label: str) -> Callable:
    """
    This evaluator measures the frequency of edges in a matching
    between two datasets that bind samples from the same class.

    Parameters
    ----------
    label: str
        .obs column entry containing labels.
    """
    return lambda a1, a2, T: 0.0 if a1 is a2 else edge_penalty(a1, a2, T, label)
