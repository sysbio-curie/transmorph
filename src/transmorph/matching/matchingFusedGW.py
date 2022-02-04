#!/usr/bin/env python3

from typing import Dict
from ot.gromov import fused_gromov_wasserstein

from scipy.spatial.distance import cdist

import numpy as np

import scanpy as sc

from .matchingABC import MatchingABC

from transmorph.TData import TData


class MatchingFusedGW(MatchingABC):
    """ """

    def __init__(
        self,
        metricM: str = "euclidean",
        metricM_kwargs: Dict = {},
        alpha: float = 0.5,
        loss: str = "square_loss",
        use_sparse: bool = True,
    ):
        MatchingABC.__init__(self, use_sparse=use_sparse)
        self.metricM = metricM
        self.metricM_kwargs = metricM_kwargs
        self.alpha = alpha
        self.loss = loss

    def _compute_di(self, x1: np.array, x2: np.array) -> np.ndarray:
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
        return M

    def _match2(self, adata1: sc.AnnData, adata2: TData) -> np.array:
        """
        Compute optimal transport plan for the FGW problem.
        Parameters
        ----------
        adata1: TData
            A dataset.
        adata2: TData
            A dataset

        Returns
        -------
        T = (xi.shape[0], xj.shape[0]) sparse array, where Tkl is the
        matching strength between xik and xjl.
        """
        n1, n2 = adata1.X.shape[0], adata2.X.shape[0]
        w1, w2 = np.ones(n1) / n1, np.ones(n2) / n2
        M = self._compute_di(adata1.X, adata2.X)
        C1 = adata1.uns["_transmorph"]["D"]
        C1 /= C1.max()
        C2 = adata2.uns["_transmorph"]["D"]
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
