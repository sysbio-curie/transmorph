#!/usr/bin/env python3

from ot.gromov import gromov_wasserstein

import numpy as np

from .matchingABC import MatchingABC
from scipy.spatial.distance import cdist
from transmorph.TData import TData
from scipy.sparse.csgraph import dijkstra
from ..utils import nearest_neighbors, pca


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
        loss: str = "square_loss",
        max_iter: int = int(1e6),
        use_sparse: bool = True,
        n_pcs: int = -1,
        geodesic: bool = True,
        n_neighbors: int = 10,
    ):
        MatchingABC.__init__(
            self, use_sparse=use_sparse, metadata_needed=["metric", "metric_kwargs"]
        )
        self.loss = loss
        self.max_iter = int(max_iter)
        self.n_pcs = n_pcs
        self.geodesic = geodesic
        self.n_neighbors = n_neighbors

    def _check_input(self, t: TData):
        if "metric_kwargs" not in t.metadata:
            t.metadata["metric_kwargs"] = {}
        if not MatchingABC._check_input(self, t):
            return False
        if self.n_pcs >= 0 and t.X.shape[1] < self.n_pcs:
            print("n_pcs >= X.shape[1]")
            return False
        return True

    def _preprocess(self, t1: TData, t2: TData):
        for t in (t1, t2):
            if "GW_distance" in t.metadata:
                continue
            X = t.X
            if self.n_pcs >= 0:
                X = pca(X, n_components=self.n_pcs)
            D = cdist(X, X, metric=t.metadata["metric"], **t.metadata["metric_kwargs"])
            if self.geodesic:
                A = nearest_neighbors(
                    X, n_neighbors=self.n_neighbors, use_nndescent=True
                )
                D = dijkstra(A.multiply(D))
                M = D[D != float("inf")].max()  # removing inf values
                D[D == float("inf")] = M
            D /= D.max()
            t.metadata["GW_distance"] = D
        return t1, t2

    def _match2(self, t1: TData, t2: TData):
        n1, n2 = t1.X.shape[0], t2.X.shape[0]
        w1, w2 = np.ones(n1) / n1, np.ones(n2) / n2
        M1 = t1.metadata["GW_distance"]
        M2 = t2.metadata["GW_distance"]
        return gromov_wasserstein(M1, M2, w1, w2, self.loss, numItermax=self.max_iter)
