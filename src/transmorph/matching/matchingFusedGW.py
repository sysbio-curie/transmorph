#!/usr/bin/env python3

from typing import Dict, Tuple
from ot.gromov import fused_gromov_wasserstein

from scipy.spatial.distance import cdist

import numpy as np

from .matchingABC import MatchingABC


class MatchingFusedGW(MatchingABC):
    """ """

    def __init__(
        self,
        metricM: str = "euclidean",
        metricM_kwargs: Dict = {},
        metricC1: str = "euclidean",
        metricC1_kwargs: Dict = {},
        metricC2: str = "euclidean",
        metricC2_kwargs: Dict = {},
        alpha: float = 0.5,
        loss: str = "square_loss",
        use_sparse: bool = True,
    ):
        MatchingABC.__init__(self, use_sparse=use_sparse)
        self.metricM = metricM
        self.metricM_kwargs = metricM_kwargs
        self.metricC1 = metricC1
        self.metricC1_kwargs = metricC1_kwargs
        self.metricC2 = metricC2
        self.metricC2_kwargs = metricC2_kwargs
        self.alpha = alpha
        self.loss = loss

    def _compute_di(self, x1: np.array, x2: np.array) -> Tuple[np.array]:
        """
        Compute cost matrices for FGW problem.
        Parameters
        ----------
        x1: np.array
            A dataset.
        x2 np.array
            A dataset

        Returns
        -------
        M, C1, C2, 3 matrices for the costs in FGW problem.
        """
        M = cdist(x1, x2, metric=self.metricM, **self.metricM_kwargs)
        C1 = cdist(x1, x1, metric=self.metricC1, **self.metricC1_kwargs)
        C2 = cdist(x2, x2, metric=self.metricC2, **self.metricC2_kwargs)
        return M, C1, C2

    def _match2(self, x1: np.array, x2: np.array) -> np.array:
        """
        Compute optimal transport plan for the FGW problem.
        Parameters
        ----------
        x1: np.array
            A dataset.
        x2: np.array
            A dataset

        Returns
        -------
        T = (xi.shape[0], xj.shape[0]) sparse array, where Tkl is the
        matching strength between xik and xjl.
        """
        n1, n2 = x1.shape[0], x2.shape[0]
        w1, w2 = np.ones(n1) / n1, np.ones(n2) / n2
        M, C1, C2 = self._compute_di(x1, x2)
        C1 /= C1.max()
        C2 /= C2.max()
        return fused_gromov_wasserstein(
            M,
            C1,
            C2,
            w1,
            w2,
            self.loss,
            self.alpha,
        )
