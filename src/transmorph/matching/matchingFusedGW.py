#!/usr/bin/env python3

from typing import Dict
from ot.gromov import fused_gromov_wasserstein

import numpy as np
import scanpy as sc
from .matchingABC import MatchingABC
from scipy.spatial.distance import cdist
from transmorph.TData import TData
from scipy.sparse.csgraph import dijkstra
from ..utils import nearest_neighbors, pca


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

    def _check_input(self, adata: sc.AnnData):
        if "metric_kwargs" not in adata.uns["_transmorph"]["matching"]:
            adata.uns["_transmorph"]["matching"]["metric_kwargs"] = {}
        if not MatchingABC._check_input(self, adata):
            return False
        if self.n_pcs >= 0 and adata.X.shape[1] < self.n_pcs:
            print("n_pcs >= X.shape[1]")
            return False
        return True

    def _preprocess(self, adata1: TData, adata2: TData):
        for adata in (adata1, adata2):
            if "GW_distance" in adata.uns["_transmorph"]:
                continue
            X = adata.X
            if self.n_pcs >= 0:
                X = pca(X, n_components=self.n_pcs)
            D = cdist(
                X,
                X,
                metric=adata.uns["_transmorph"]["matching"]["metric"],
                **adata.uns["_transmorph"]["matching"]["metric_kwargs"]
            )
            if self.geodesic:
                A = nearest_neighbors(
                    X, n_neighbors=self.n_neighbors, use_nndescent=True
                )
                D = dijkstra(A.multiply(D))
                M = D[D != float("inf")].max()  # removing inf values
                D[D == float("inf")] = M
            D /= D.max()
            adata.uns["_transmorph"]["GW_distance"] = D
        return adata1, adata2

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
        C1 = adata1.uns["_transmorph"]["GW_distance"]
        C1 /= C1.max()
        C2 = adata2.uns["_transmorph"]["GW_distance"]
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
