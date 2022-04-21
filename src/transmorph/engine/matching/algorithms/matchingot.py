#!/usr/bin/env python3

import numpy as np

from ot import emd
from ot.bregman import sinkhorn_stabilized
from ot.partial import partial_wasserstein
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from typing import Dict, List, Literal, Optional

from .. import Matching, _TypeMatchingSet
from ...profiler import profile_method
from ...traits import UsesCommonFeatures


class MatchingOT(Matching, UsesCommonFeatures):
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

    common_features_mode: Literal["pairwise", "total"]
        Uses pairwise common features, or total common features. Use "total"
        for a small number of datasets, and "pairwise" if the features
        intersection is too small.
    """

    def __init__(
        self,
        optimizer: Literal["emd", "sinkhorn", "partial"] = "emd",
        metric: str = "sqeuclidean",
        metric_kwargs: Optional[Dict] = None,
        common_features_mode: Literal["pairwise", "total"] = "pairwise",
        sinkhorn_epsilon: Optional[float] = None,
        partial_transport_mass: Optional[float] = None,
        partial_n_dummies: Optional[int] = None,
        max_iter: int = int(1e6),
    ):
        Matching.__init__(self, str_identifier="OT")
        UsesCommonFeatures.__init__(self, mode=common_features_mode)
        assert optimizer in (
            "emd",
            "sinkhorn",
            "partial",
        ), f"Unknown optimizer: {optimizer}."
        self.optimizer = optimizer
        self.metric = metric
        self.metric_kwargs = {} if metric_kwargs is None else metric_kwargs
        self.max_iter = int(max_iter)
        if sinkhorn_epsilon is not None and optimizer != "sinkhorn":
            self.warn(
                f"Setting sinkhorn epsilon has no effect for optimizer {optimizer}."
            )
        if (
            partial_n_dummies is not None or partial_transport_mass is not None
        ) and optimizer != "partial":
            self.warn(
                "Setting partial_n_dummies or partial_transport_mass "
                f"has no effect for optimizer {optimizer}."
            )
        if sinkhorn_epsilon is None:
            sinkhorn_epsilon = 1e-2
        if partial_transport_mass is None:
            partial_transport_mass = 1.0
        if partial_n_dummies is None:
            partial_n_dummies = 1
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.partial_transport_mass = partial_transport_mass
        self.partial_n_dummies = partial_n_dummies

    @profile_method
    def fit(self, datasets: List[np.ndarray]) -> _TypeMatchingSet:
        """
        Computes OT between pairs of datasets with the right optimizer.
        """
        ndatasets = len(datasets)
        kwargs = {}
        if self.optimizer == "emd":
            optimizer = emd
        elif self.optimizer == "sinkhorn":
            optimizer = sinkhorn_stabilized
            kwargs = {"reg": self.sinkhorn_epsilon}
        elif self.optimizer == "partial":
            optimizer = partial_wasserstein
            kwargs = {
                "m": self.partial_transport_mass,
                "nb_dummies": self.partial_n_dummies,
            }
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        results: _TypeMatchingSet = {}
        for i in range(ndatasets):
            for j in range(i + 1, ndatasets):
                Xi, Xj = datasets[i], datasets[j]
                ni, nj = Xi.shape[0], Xj.shape[0]
                wi, wj = np.ones(ni) / ni, np.ones(nj) / nj
                Xj, Xj = self.slice_features(X1=Xi, X2=Xj, idx_1=i, idx_2=j)
                M = cdist(Xi, Xj, metric=self.metric, **self.metric_kwargs)
                M /= M.max()
                Tij = optimizer(a=wi, b=wj, M=M, numItermax=self.max_iter, **kwargs)
                results[i, j] = csr_matrix(Tij)
                results[j, i] = csr_matrix(Tij.T)
        return results
