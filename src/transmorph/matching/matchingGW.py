#!/usr/bin/env python3

from ot.gromov import gromov_wasserstein
from scipy.spatial.distance import cdist
from typing import Union, Callable

import numpy as np

from .matchingABC import MatchingABC


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
        # geodesic: bool = True,
        metric: Union[str, Callable] = "sqeuclidean",
        metric_kwargs: dict = {},
        loss: str = "square_loss",
        max_iter: int = int(1e6),
        use_sparse: bool = True,
    ):
        MatchingABC.__init__(self, use_sparse=use_sparse)
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.loss = loss
        self.max_iter = int(max_iter)

    def _match2(self, x1, x2):
        n1, n2 = x1.shape[0], x2.shape[0]
        w1, w2 = np.ones(n1) / n1, np.ones(n2) / n2
        M1 = cdist(x1, x1, metric=self.metric, **self.metric_kwargs)
        M1 /= M1.max()
        M2 = cdist(x2, x2, metric=self.metric2, **self.metric2_kwargs)
        M2 /= M2.max()
        return gromov_wasserstein(M1, M2, w1, w2, self.loss)
