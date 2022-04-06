#!/usr/bin/env python3

import numpy as np

from anndata import AnnData
from scipy.spatial.distance import cdist
from typing import Callable, Dict, Optional, Union
from ot.gromov import gromov_wasserstein

from .matchingABC import MatchingABC
from ..subsampling.subsamplingABC import SubsamplingABC
from ..utils.anndata_interface import isset_info, set_info, get_info


class MatchingGW(MatchingABC):
    """
    Gromov-Wasserstein-based matching. Embeds the gromov_wasserstein
    method from POT:

        https://github.com/PythonOT/POT

    Gromov-Wasserstein(GW) computes a transport plan between two distributions,
    and does not require them to be defined in the same space. It rather use
    relative topology of each distribution in its own metric space to define a
    cost that assumes similar locations to have similar relative positions with
    respect to the other regions. This combinatorial cost is typically more
    expansive than the optimal transport alternative, but comes very handy when
    a ground cost is difficult (or impossible) to compute between
    distributions.

    Parameters
    ----------
    metric: str or callable, default = "sqeuclidean"
        Scipy-compatible metric to use for datasets in which metric
        is not specified as a backup.

    metric_kwargs: dict, default = {}
        Additional metric parameters for backup metric.

    GW_loss: str, default = "square_loss"
        Loss to use in the Gromov-Wasserstein problem. Valid options
        are "square_loss", "kl_loss".

    max_iter: int, default = 1e6
        Maximum number of iterations to solve the optimization problem.

    subsampling: SubsamplingABC, default = None
        Subsampling scheme to apply before computing the matching,
        can be very helpful when dealing with large datasets.

    References
    ----------
    [1] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    [2] Mémoli, Facundo. Gromov–Wasserstein distances and the
        metric approach to object matching. Foundations of computational
        mathematics 11.4 (2011): 417-487.
    """

    def __init__(
        self,
        metric: Union[str, Callable] = "sqeuclidean",
        metric_kwargs: Optional[Dict] = None,
        GW_loss: str = "square_loss",
        max_iter: int = int(1e6),
        subsampling: Optional[SubsamplingABC] = None,
    ):

        super().__init__(
            metadata_keys=["metric", "metric_kwargs"], subsampling=subsampling
        )
        self.loss = GW_loss
        self.metric = metric
        self.metric_kwargs = {} if metric_kwargs is None else metric_kwargs
        self.max_iter = int(max_iter)

    def _check_input(self, adata: AnnData, dataset_key: str = ""):
        """
        Adds some default metric information if needed.
        """
        if not isset_info(adata, "metric"):
            set_info(adata, "metric", self.metric)
        if not isset_info(adata, "metric_kwargs"):
            set_info(adata, "metric_kwargs", self.metric_kwargs)
        return super()._check_input(adata, dataset_key)

    def _match2(self, adata1: AnnData, adata2: AnnData) -> np.ndarray:
        """
        Compute optimal transport plan for the GW problem.

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
        X1 = MatchingABC.to_match(adata1)
        X2 = MatchingABC.to_match(adata2)

        metric_1 = get_info(adata1, "metric")
        metric_1_kwargs = get_info(adata1, "metric_kwargs")
        C1 = cdist(X1, X1, metric_1, **metric_1_kwargs)
        C1 /= C1.max()

        metric_2 = get_info(adata2, "metric")
        metric_2_kwargs = get_info(adata2, "metric_kwargs")
        C2 = cdist(X2, X2, metric_2, **metric_2_kwargs)
        C2 /= C2.max()

        return gromov_wasserstein(C1, C2, w1, w2, self.loss, numItermax=self.max_iter)
