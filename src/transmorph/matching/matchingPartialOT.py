#!/usr/bin/env python3
#
import numpy as np

from anndata import AnnData
from ot.partial import partial_wasserstein
from scipy.spatial.distance import cdist
from typing import Optional

from .matchingABC import MatchingABC
from ..subsampling.subsamplingABC import SubsamplingABC


class MatchingPartialOT(MatchingABC):
    """
    Partial Optimal Transport matching, embedding the
    method from POT:

        https://github.com/PythonOT/POT

    ot.emd solves the problem of optimal transport where only
    a portion of mass needs to be displaced. It is quite useful
    when dealing with class-specific objects, for instance cell
    types specific to a dataset that we do not want to match.

    Parameters
    ----------
    transport_mass: float, default = 1.0
        Fraction of mass to move, an estimator of the common
        samples between datasets. If transport_mass == 1.0, equivalent
        to OT formulation.

    n_dummies: int, default = 1
        Number of dummy points to use during partial OT. Increasing
        this number may yield additional stability.

    metric: str or callable, default = "sqeuclidean"
        Scipy-compatible metric.

    metric_kwargs: dict, default = {}
        Additional metric parameters.

    max_iter: int, default = 1e6
        Maximum number of iterations to solve the optimization problem.

    subsampling: SubsamplingABC, default = None
        Subsampling scheme to apply before computing the matching,
        can be very helpful when dealing with large datasets.
    """

    def __init__(
        self,
        transport_mass: float = 1.0,
        n_dummies: int = 1,
        metric: str = "sqeuclidean",
        metric_kwargs: dict = {},
        max_iter: int = int(1e6),
        subsampling: Optional[SubsamplingABC] = None,
    ):
        super().__init__(metadata_keys=[], subsampling=subsampling)
        self.transport_mass = transport_mass
        self.n_dummies = n_dummies
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.max_iter = int(max_iter)

    def _match2(self, adata1: AnnData, adata2: AnnData) -> np.ndarray:
        X1 = self.to_match(adata1)
        X2 = self.to_match(adata2)
        n1, n2 = X1.shape[0], X2.shape[0]
        w1, w2 = np.ones(n1) / n1, np.ones(n2) / n2
        M = cdist(X1, X2, metric=self.metric, **self.metric_kwargs)
        M /= M.max()
        transport_mass = min(self.transport_mass, min(w1.sum(), w2.sum()))
        return partial_wasserstein(
            w1,
            w2,
            M,
            m=transport_mass,
            nb_dummies=self.n_dummies,
            numItermax=self.max_iter,
        )
