#!/usr/bin/env python3

from ot.gromov import entropic_gromov_wasserstein
from typing import Union, Callable

import numpy as np

from .matchingABC import MatchingABC
from src.transmorph.TData import TData


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

    References
    ----------
    [1] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """

    def __init__(
        self,
        geodesic: bool = True,
        metric: Union[str, Callable] = "sqeuclidean",
        metric_kwargs: dict = {},
        epsilon: float = 1e-2,
        loss: str = "square_loss",
        max_iter: int = int(1e6),
        use_sparse: bool = True,
    ):
        MatchingABC.__init__(self, use_sparse=use_sparse)
        self.geodesic = geodesic
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.epsilon = epsilon
        self.loss = loss
        self.max_iter = int(max_iter)

    def _match2(self, t1: TData, t2: TData):
        n1, n2 = t1.X.shape[0], t2.X.shape[0]
        w1, w2 = np.ones(n1) / n1, np.ones(n2) / n2
        M1 = t1.D.copy()
        M1 /= M1.max()
        M2 = t2.D.copy()
        M2 /= M2.max()
        return entropic_gromov_wasserstein(
            M1, M2, w1, w2, self.loss, self.epsilon, max_iter=self.max_iter
        )
