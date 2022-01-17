#!/usr/bin/env python3

from ot.gromov import entropic_gromov_wasserstein
from scipy.spatial.distance import cdist
from typing import Union, Callable

import numpy as np

from .matchingABC import MatchingABC


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
        geodesic: bool = True,
        metric: Union[str, Callable] = "sqeuclidean",
        metric_kwargs: dict = {},
        epsilon: float = 1e-2,
        loss: str = "square_loss",
        max_iter: int = int(1e6),
        use_sparse: bool = True,
        low_cut: float = 1e-3,
    ):
        MatchingABC.__init__(self, use_sparse=use_sparse)
        self.geodesic = geodesic
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.epsilon = epsilon
        self.loss = loss
        self.max_iter = int(max_iter)
        self.low_cut = low_cut

    def _match2(self, x1: np.ndarray, x2: np.ndarray):
        n1, n2 = x1.shape[0], x2.shape[0]
        w1, w2 = np.ones(n1) / n1, np.ones(n2) / n2
        M1 = cdist(x1, x1, metric=self.metric, **self.metric_kwargs)
        M1 /= M1.max()
        M2 = cdist(x2, x2, metric=self.metric, **self.metric_kwargs)
        M2 /= M2.max()
        T = entropic_gromov_wasserstein(
            M1, M2, w1, w2, self.loss, self.epsilon, max_iter=self.max_iter
        )
        if self.use_sparse:
            low_cut = self.low_cut / n1
            T = T * (T > low_cut)
        return T
