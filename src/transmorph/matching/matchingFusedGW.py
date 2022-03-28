#!/usr/bin/env python3

from typing import Dict
from ot.gromov import fused_gromov_wasserstein

import numpy as np

from ..subsampling.subsamplingABC import SubsamplingABC
from ..subsampling import SubsamplingKeepAll
from ..utils.anndata_interface import (
    get_attribute,
    isset_attribute,
    set_attribute,
)
from .matchingABC import MatchingABC
from scipy.spatial.distance import cdist
from anndata import AnnData


class MatchingFusedGW(MatchingABC):
    """
    Fused Gromov-Wasserstein-based matching. Embeds the
    ot.gromov.fused_gromov_wasserstein method from POT:

        https://github.com/PythonOT/POT

    It computes a combination of Gromov-Wasserstein and Optimal
    Transport, weighted by a coefficient alpha.
    Both datasets need to be in the same
    space in order to compute a cost matrix.

    Parameters
    ----------
    OT_metric: str or callable, default = "sqeuclidean"
        Scipy-compatible metric for optimal transport cost matrix.

    OT_metric_kwargs: dict, default = {}
        Additional metric parameters for OT_metric.

    alpha: float, default = 0.5
        Ratio between optimal transport and Gromov-Wasserstein terms
        in the optimization problem.

    max_iter: int, default = 1e6
        Maximum number of iterations to solve the optimization problem.
    """

    def __init__(
        self,
        OT_metric: str = "sqeuclidean",
        OT_metric_kwargs: Dict = {},
        alpha: float = 0.5,
        GW_loss: str = "square_loss",
        subsampling: SubsamplingABC = SubsamplingKeepAll(),
    ):
        super().__init__(
            metadata_keys=["metric", "metric_kwargs"], subsampling=subsampling
        )
        self.OT_metric = OT_metric
        self.OT_metric_kwargs = OT_metric_kwargs
        self.alpha = alpha
        self.GW_loss = GW_loss

    def _check_input(self, adata: AnnData, dataset_key: str = "") -> None:
        """
        Adds some default metric information if needed.
        """
        if not isset_attribute(adata, "metric"):
            set_attribute(adata, "metric", self.OT_metric)
        if not isset_attribute(adata, "metric_kwargs"):
            set_attribute(adata, "metric_kwargs", self.OT_metric_kwargs)
        return super()._check_input(adata, dataset_key)

    def _match2(self, adata1: AnnData, adata2: AnnData) -> np.ndarray:
        """
        Compute optimal transport plan for the FGW problem.

        Parameters
        ----------
        adata1: AnnData
            A dataset.
        adata2: AnnData
            A dataset

        Returns
        -------
        T = (xi.shape[0], xj.shape[0]) sparse array, where Tkl is the
        matching strength between xik and xjl.
        """
        n1, n2 = adata1.X.shape[0], adata2.X.shape[0]
        w1, w2 = np.ones(n1) / n1, np.ones(n2) / n2
        X1 = self.to_match(adata1)
        X2 = self.to_match(adata2)

        M = cdist(X1, X2, metric=self.OT_metric, *self.OT_metric_kwargs)
        M /= M.max()

        metric_1 = get_attribute(adata1, "metric")
        metric_1_kwargs = get_attribute(adata1, "metric_kwargs")
        C1 = cdist(X1, X1, metric_1, **metric_1_kwargs)
        C1 /= C1.max()

        metric_2 = get_attribute(adata2, "metric")
        metric_2_kwargs = get_attribute(adata2, "metric_kwargs")
        C2 = cdist(X2, X2, metric_2, **metric_2_kwargs)
        C2 /= C2.max()

        return fused_gromov_wasserstein(
            M,
            C1,
            C2,
            w1,
            w2,
            self.GW_loss,
            self.alpha,
        )
