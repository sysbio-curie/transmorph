#!/usr/bin/env python3

import numpy as np

from anndata import AnnData
from ot.bregman import sinkhorn_stabilized
from scipy.spatial.distance import cdist
from typing import Dict, Optional


from .matchingABC import MatchingABC
from ..subsampling.subsamplingABC import SubsamplingABC


class MatchingSinkhorn(MatchingABC):
    """
    Approximate optimal transport-based matching [1]. Embeds the
    method from POT sinkhorn_stabilized [2]:

        https://github.com/PythonOT/POT

    This solves approximately the optimal transport problem
    using an entropic regularization.

    Parameters
    ----------
    metric: str or callable, default = "sqeuclidean"
        Scipy-compatible metric.

    metric_kwargs: dict, default = {}
        Additional metric parameters.

    max_iter: int, default = 1e6
        Maximum number of iterations to solve the optimization problem.

    low_cut: bool, default = True
        Entropic GW solver usually returns not sparse matchings with
        a lot of very low residuals. Using a low cut filter allows to
        sparsify returned values, with only a small impact on quality
        but huge computational downstream benefits. Turn this off if you
        want to keep the non-sparse matching.

    low_cut_threshold: float, default = 0.001
        If low_cut = True, then all values in the final matrix lesser
        than 1 / (n1 * n2) * low_cut_threshold are discarded (where
        n1 and n2 are the number of points in each dataset).

    subsampling: SubsamplingABC, default = None
        Subsampling scheme to apply before computing the matching,
        can be very helpful when dealing with large datasets.

    References
    ----------
    [1] TODO Cuturi paper on sinkhorn
    [2] TODO Paper on sinkhorn stabilized
    """

    def __init__(
        self,
        metric: str = "sqeuclidean",
        metric_kwargs: Optional[Dict] = None,
        epsilon: float = 1e-2,
        max_iter: int = int(5e2),
        low_cut: bool = True,
        low_cut_thr: float = 1e-3,
        subsampling: Optional[SubsamplingABC] = None,
    ):
        super().__init__(metadata_keys=[], subsampling=subsampling)
        self.metric = metric
        self.metric_kwargs = {} if metric_kwargs is None else metric_kwargs
        self.epsilon = epsilon
        self.max_iter = int(max_iter)
        self.low_cut = low_cut
        self.low_cut_thr = low_cut_thr

    def _match2(self, adata1: AnnData, adata2: AnnData) -> np.ndarray:
        X1 = MatchingABC.to_match(adata1)
        X2 = MatchingABC.to_match(adata2)
        n1, n2 = X1.shape[0], X2.shape[0]
        w1, w2 = np.ones(n1) / n1, np.ones(n2) / n2
        M = cdist(X1, X2, metric=self.metric, **self.metric_kwargs)
        M /= M.max()
        T = sinkhorn_stabilized(w1, w2, M, self.epsilon, numItermax=self.max_iter)
        low_cut = 1.0 / (T.shape[0] * T.shape[1]) * self.low_cut_thr
        T = T * (T > low_cut)
        T = T / T.sum(axis=1, keepdims=True) / T.shape[1]
        return T
