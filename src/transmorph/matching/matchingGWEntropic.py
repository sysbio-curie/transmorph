#!/usr/bin/env python3

import numpy as np

from anndata import AnnData
from ot.gromov import entropic_gromov_wasserstein
from scipy.spatial.distance import cdist
from typing import Optional

from .matchingABC import MatchingABC
from ..subsampling.subsamplingABC import SubsamplingABC
from ..utils.anndata_interface import isset_info, set_info, get_info


class MatchingGWEntropic(MatchingABC):
    """
    Entropic Gromov-Wasserstein-based matching. Embeds the
    entropic_gromov_wasserstein method from POT:

        https://github.com/PythonOT/POT

    Gromov-Wasserstein(GW) computes a transport plan between two distributions,
    and does not require them to be defined in the same space. It rather use
    relative topology of each distribution in its own metric space to define a
    cost that assumes similar locations to have similar relative positions with
    respect to the other regions. This combinatorial cost is typically more
    expansive than the optimal transport alternative, but comes very handy when
    a ground cost is difficult (or impossible) to compute between
    distributions.

    Here, we use an entropy regularized variant, easier to optimize at a cost
    of a regularization parameter. Furthermore, this approach tends to alter
    the matching sparse structure.

    Parameters
    ----------
    metric: str or callable, default = "sqeuclidean"
        Scipy-compatible metric to use for datasets in which metric
        is not specified as a backup.

    metric_kwargs: dict, default = {}
        Additional metric parameters for backup metric.

    epsilon: float, default = 0.01
        Entropy regularization coefficient in the approximate problem.

    GW_loss: str, default = "square_loss"
        Either "square_loss" or "kl_loss". Passed to gromov_wasserstein for the
        optimization.

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
    [1] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """

    def __init__(
        self,
        metric: str = "sqeuclidean",
        metric_kwargs: dict = {},
        epsilon: float = 1e-2,
        GW_loss: str = "square_loss",
        max_iter: int = int(1e6),
        low_cut: bool = True,
        low_cut_thr: float = 1e-3,
        subsampling: Optional[SubsamplingABC] = None,
    ):
        super().__init__(
            metadata_keys=["metric", "metric_kwargs"], subsampling=subsampling
        )
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.epsilon = epsilon
        self.loss = GW_loss
        self.max_iter = int(max_iter)
        self.low_cut = low_cut
        self.low_cut_thr = low_cut_thr

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
        Compute approximate optimal transport plan for the GW problem.

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

        metric_1 = get_info(adata1, "metric")
        metric_1_kwargs = get_info(adata1, "metric_kwargs")
        C1 = cdist(X1, X1, metric_1, **metric_1_kwargs)
        C1 /= C1.max()

        metric_2 = get_info(adata2, "metric")
        metric_2_kwargs = get_info(adata2, "metric_kwargs")
        C2 = cdist(X2, X2, metric_2, **metric_2_kwargs)
        C2 /= C2.max()

        T = entropic_gromov_wasserstein(
            C1, C2, w1, w2, self.loss, self.epsilon, max_iter=self.max_iter
        )
        low_cut = 1.0 / (T.shape[0] * T.shape[1]) * self.low_cut_thr
        T = T * (T > low_cut)
        T = T / T.sum(axis=1, keepdims=True) / T.shape[1]
        return T
