#!/usr/bin/env python3


import numpy as np

from anndata import AnnData
from ot import emd
from scipy.spatial.distance import cdist
from typing import Optional

from .matchingABC import MatchingABC
from ..subsampling.subsamplingABC import SubsamplingABC


class MatchingEMD(MatchingABC):
    """
    Earth Mover's Distance-based matching. Embeds the ot.emd
    method from POT:

        https://github.com/PythonOT/POT

    ot.emd solves exactly the earth mover's distance problem using
    a C-accelerated backend. Both datasets need to be in the same
    space in order to compute a cost matrix.

    Parameters
    ----------
    metric: str or callable, default = "sqeuclidean"
        Scipy-compatible metric.

    metric_kwargs: dict, default = {}
        Additional metric parameters.

    subsampling: SubsamplingABC, default = None
        Subsampling scheme to apply before computing the matching,
        can be very helpful when dealing with large datasets.

    max_iter: int, default = 1e6
        Maximum number of iterations to solve the optimization problem.
    """

    def __init__(
        self,
        metric: str = "sqeuclidean",
        metric_kwargs: dict = {},
        subsampling: Optional[SubsamplingABC] = None,
        max_iter: int = int(1e6),
    ):
        super().__init__(metadata_keys=[], subsampling=subsampling)
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.max_iter = int(max_iter)

    def _match2(self, adata1: AnnData, adata2: AnnData) -> np.ndarray:
        """
        Simply gathers datasets and compute exact optimal transport.
        """
        X1, X2 = self.to_match(adata1), self.to_match(adata2)
        n1, n2 = X1.shape[0], X2.shape[0]
        w1, w2 = np.ones(n1) / n1, np.ones(n2) / n2
        M = cdist(X1, X2, metric=self.metric, **self.metric_kwargs)
        M /= M.max()
        return emd(w1, w2, M, numItermax=self.max_iter)
