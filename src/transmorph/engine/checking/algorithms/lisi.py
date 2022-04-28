#!/usr/bin/env python3

import numpy as np

from typing import List, Optional

from ..checking import Checking
from ....stats.lisi import compute_lisi


class LISI(Checking):
    """
    LISI statistic measures how heterogeneous a sample neighborhood
    is for a certain label. Is is notably used in the Harmony
    integration pipeline to measure how well integrated datasets
    are. Here, we use it to measure how many batches are found
    on average in each sample's neighborhood.

    Parameters
    ----------
    threshold: float, default = 0.5
        Minimum average heterogeneity to reach to accept
        an integration, after dividing by the number of
        datasets.

    perplexity: float, default = 30.0
        Scales the neighborhood size to consider.
    """

    def __init__(self, threshold: float = 0.5, perplexity: float = 30.0):
        Checking.__init__(self, str_identifier="LISI")
        self.threshold = threshold
        self.perplexity = perplexity
        self.score: Optional[float] = None

    def check(self, datasets: List[np.ndarray]) -> bool:
        """
        Computes LISI after concatenating datasets, then
        compares with threshold.
        """
        X_all = np.concatenate(datasets, axis=0)
        labels = np.array(sum([[i] * X.shape[0] for i, X in enumerate(datasets)], []))
        lisi = compute_lisi(X_all, labels, self.perplexity)
        self.score = lisi.mean()
        return self.score >= self.threshold