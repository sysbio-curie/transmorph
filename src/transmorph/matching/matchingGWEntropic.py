#!/usr/bin/env python3

from ot.gromov import entropic_gromov_wasserstein
from typing import Union, Callable

import numpy as np
from anndata import AnnData
from scipy.spatial.distance import cdist
from .matchingABC import MatchingABC
from ..utils.anndata_interface import isset_attribute, set_attribute, get_attribute


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
        Scipy-compatible metric.

    metric_kwargs: dict, default = {}
        Additional metric parameters.

    loss: str, default = "square_loss"
        Either "square_loss" or "kl_loss". Passed to gromov_wasserstein for the
        optimization.

    max_iter: int, default = 1e6
        Maximum number of iterations to solve the optimization problem.

    use_sparse: boolean, default = True
        Save matching as sparse matrices.

    low_cut:

    References
    ----------
    [1] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """

    def __init__(
        self,
        metric: Union[str, Callable] = "sqeuclidean",
        metric_kwargs: dict = {},
        epsilon: float = 1e-2,
        loss: str = "square_loss",
        max_iter: int = int(1e6),
        low_cut: bool = True,
        low_cut_thr: float = 1e-3,
    ):
        super().__init__(metadata_keys=["metric", "metric_kwargs"])
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.epsilon = epsilon
        self.loss = loss
        self.max_iter = int(max_iter)
        self.low_cut = low_cut
        self.low_cut_thr = low_cut_thr

    def _check_input(self, adata: AnnData, dataset_key: str = ""):
        """
        Adds some default metric information if needed.
        """
        if not isset_attribute(adata, "metric"):
            set_attribute(adata, "metric", self.metric)
        if not isset_attribute(adata, "metric_kwargs"):
            set_attribute(adata, "metric_kwargs", self.metric_kwargs)
        return super()._check_input(adata, dataset_key)

    def _match2(self, adata1: AnnData, adata2: AnnData):
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

        metric_1 = get_attribute(adata1, "metric")
        metric_1_kwargs = get_attribute(adata1, "metric_kwargs")
        C1 = cdist(X1, X1, metric_1, **metric_1_kwargs)
        C1 /= C1.max()

        metric_2 = get_attribute(adata2, "metric")
        metric_2_kwargs = get_attribute(adata2, "metric_kwargs")
        C2 = cdist(X2, X2, metric_2, **metric_2_kwargs)
        C2 /= C2.max()

        T = entropic_gromov_wasserstein(
            C1, C2, w1, w2, self.loss, self.epsilon, max_iter=self.max_iter
        )
        low_cut = self.low_cut / n1
        T = T * (T > low_cut)
        return T
