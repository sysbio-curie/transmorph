#!/usr/bin/env python3

from ot import emd 
from ot.bregman import sinkhorn_stabilized
from ot.gromov import (
    entropic_gromov_wasserstein,
    gromov_wasserstein,
    fused_gromov_wasserstein,
)
from scipy.spatial.distance import cdist

import numpy as np

from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix
from typing import Union


class MatchingABC(ABC):
    """
    A matching is a class containing a function match(x1, ..., xn), able
    to predict matching between dataset samples (possibly fuzzy). Any class
    implementing Matching must implement a

        _match2(self, x1: np.ndarray, x2: np.ndarray)

    method, returning a possibly sparse T = (x1.shape[0], x2.shape[0]) array
    where T[i, j] = prob(x1_i matches x2_j).

    Parameters
    ----------
    use_sparse: boolean
        Save matchings as sparse matrices.

    Attributes
    ----------
    fitted: boolean
        Is true if match() method has been successfully exectuted.

    matchings: list of arrays
        After calling match(x0, ..., xn), matching[i(i-1)/2+j] contains the
        matching between xi and xj (with i > j).
    """
    @abstractmethod
    def __init__(
            self,
            use_sparse: bool=True
    ):
        self.fitted = False
        self.matchings = []
        self.use_sparse = use_sparse    
        

    @abstractmethod
    def _match2(
            self,
            x1: np.ndarray,
            x2: np.ndarray
    ) -> np.ndarray:
        pass


    def get(
            self,
            i: int,
            j: int,
            normalize: bool=False
    ) -> Union[np.ndarray, csr_matrix]:
        """
        Return the matching between datasets i and j. Throws an error
        if Matching is not fitted, or if i == j.

        Parameters
        ----------
        i: int
            Index of the source dataset (samples in rows).

        j: int
            Index of the reference dataset (samples in columns).

        normalize: bool
            Normalize each row to one.

        Returns
        -------
        T = (xi.shape[0], xj.shape[0]) as a numpy array or a CSR
        matrix depending on self.use_sparse, where Tkl is the
        matching strength between xik and xjl.
        """
        assert self.fitted, \
            "Error: Matching not fitted, call match() first."
        assert i != j, \
            "Error: i = j."
        transpose = i < j
        if transpose:
            i, j = j, i
        index = int(i * (i - 1) / 2 + j)
        assert index < len(self.matchings), \
            f"Index ({i}, {j}) out of bounds."
        T = self.matchings[index]
        if transpose:
            T = T.T
        if normalize:
            return T / T.sum(axis=1)
        return T


    def match(self, *datasets: np.ndarray):
        """
        Matches all pairs of different datasets together. Returns results
        in a dictionary, where d[i,j] is the matching between datasets i
        and j represented as a (ni, nj) numpy array -- possibly fuzzy.

        Parameters:
        -----------
        *datasets: tuple of arrays
            Tuple of at least two datasets. 
        """
        self.fitted = False
        self.matchings = []
        nd = len(datasets)
        assert nd > 1, "Error: at least 2 datasets required."
        for i in range(nd):
            di = datasets[i]
            for j in range(i):
                matching = self._match2(di, datasets[j])
                if self.use_sparse:
                    matching = csr_matrix(matching)
                self.matchings.append(matching)
        self.fitted = True
        return self.matchings


class MatchingEMD(MatchingABC):
    """
    Earth Mover's Distance-based matching. Embeds the ot.emd
    method from POT:

        https://github.com/PythonOT/POT

    ot.emd solves exactly the earth mover's distance problem using
    a C-accelerated backend. Both datasets need to be in the same
    space in order to compute a cost matrix.
    TODO: allow the user to define a custom callable metric?

    Parameters
    ----------
    metric: str or callable, default = "sqeuclidean"
        Scipy-compatible metric.

    metric_kwargs: dict, default = {}
        Additional metric parameters.

    max_iter: int, default = 1e6
        Maximum number of iterations to solve the optimization problem.

    use_sparse: boolean, default = True
        Save matching as sparse matrices.
    """
    def __init__(
            self,
            metric="sqeuclidean",
            metric_kwargs={},
            max_iter=1e6,
            use_sparse=True
    ):
        MatchingABC.__init__(self, use_sparse=use_sparse)
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.max_iter = int(max_iter)


    def _match2(self, x1, x2):
        n1, n2 = x1.shape[0], x2.shape[0]
        w1, w2 = np.ones(n1) / n1, np.ones(n2) / n2
        M = cdist(
            x1,
            x2,
            metric=self.metric,
            **self.metric_kwargs
        )
        M /= M.max()
        return emd(w1, w2, M, numItermax=self.max_iter)


class MatchingSinkhorn(MatchingABC):
    """
    Entropic optimal transport-based matching. Embeds the sinkhorn_stabilized
    method from POT:

        https://github.com/PythonOT/POT

    This approximately computes the optimal transport plan between both datasets
    using entropy regularization [1], which makes the problem strongly convex.
    Furthermore, additional robustness is achieved through logarithmic
    stabilization [2]. Both datasets need to be in the same
    space in order to compute a cost matrix. This can typically scale to larger
    dataset, at a cost of a regularization parameter and less sparsity in the
    final matching.
    TODO: allow the user to define a custom callable metric?

    Parameters
    ----------
    metric: str or callable, default = "sqeuclidean"
        Scipy-compatible metric.

    metric_kwargs: dict, default = {}
        Additional metric parameters.

    epsilon: float, default = 1e-2
        Entropy regularization parameter.

    max_iter: int, default = 1e6
        Maximum number of iterations to solve the optimization problem.

    use_sparse: boolean, default = True
        Save matching as sparse matrices.

    References
    ----------
    [1] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of
        Optimal Transport, Advances in Neural Information Processing
        Systems (NIPS) 26, 2013
    [2] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms
        for Entropy Regularized Transport Problems.
        arXiv preprint arXiv:1610.06519.
    """
    def __init__(
            self,
            metric="sqeuclidean",
            metric_kwargs={},
            epsilon=1e-2,
            max_iter=5e2,
            use_sparse=True
    ):
        MatchingABC.__init__(self, use_sparse=use_sparse)
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.epsilon = epsilon
        self.max_iter = int(max_iter)


    def _match2(self, x1, x2):
        n1, n2 = x1.shape[0], x2.shape[0]
        w1, w2 = np.ones(n1) / n1, np.ones(n2) / n2
        M = cdist(
            x1,
            x2,
            metric=self.metric,
            **self.metric_kwargs
        )
        M /= M.max()
        return sinkhorn_stabilized(
            w1,
            w2,
            M,
            self.epsilon,
            numItermax=self.max_iter
        )


class MatchingGW(MatchingABC):
    """
    Gromov-Wasserstein-based matching. Embeds the gromov_wasserstein
    method from POT:

        https://github.com/PythonOT/POT

    Gromov-Wasserstein (GW) computes a transport plan between two distributions,
    and does not require them to be defined in the same space. It rather use
    relative topology of each distribution in its own metric space to define a
    cost that assumes similar locations to have similar relative positions with
    respect to the other regions. This combinatorial cost is typically more
    expansive than the optimal transport alternative, but comes very handy when
    a ground cost is difficult (or impossible) to compute between distributions.

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
            metric="sqeuclidean",
            metric_kwargs={},
            loss="square_loss",
            max_iter=1e6,
            use_sparse=True
    ):
        MatchingABC.__init__(self, use_sparse=use_sparse)
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.loss = loss
        self.max_iter = int(max_iter)


    def _match2(self, x1, x2):
        n1, n2 = x1.shape[0], x2.shape[0]
        w1, w2 = np.ones(n1) / n1, np.ones(n2) / n2
        M1 = cdist(
            x1,
            x1,
            metric=self.metric,
            **self.metric_kwargs
        )
        M1 /= M1.max()
        M2 = cdist(
            x2,
            x2,
            metric=self.metric2,
            **self.metric2_kwargs
        )
        M2 /= M2.max()
        return gromov_wasserstein(
            M1,
            M2,
            w1,
            w2,
            self.loss
        )


class MatchingGWEntropic(MatchingABC):
    """

    """
    def __init__(
            self,
            metric="sqeuclidean",
            metric_kwargs={},
            metric2=None,
            metric2_kwargs={},
            epsilon=1e-2,
            loss="square_loss",
            max_iter=1e6,
            use_sparse=True
    ):
        Matching.__init__(self, use_sparse=use_sparse)
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        if metric2 is None:
            self.metric2 = self.metric
            self.metric2_kwargs = self.metric_kwargs
        else:
            self.metric2 = metric2
            self.metric2_kwargs = metric2_kwargs
        self.epsilon = epsilon
        self.loss = loss
        self.max_iter = int(max_iter)


    def _match2(self, x1, x2):
        n1, n2 = x1.shape[0], x2.shape[0]
        w1, w2 = np.ones(n1) / n1, np.ones(n2) / n2
        M1 = cdist(
            x1,
            x1,
            metric=self.metric,
            **self.metric_kwargs
        )
        M1 /= M1.max()
        M2 = cdist(
            x2,
            x2,
            metric=self.metric2,
            **self.metric2_kwargs
        )
        M2 /= M2.max()
        return entropic_gromov_wasserstein(
            M1,
            M2,
            w1,
            w2,
            self.loss,
            self.epsilon,
            max_iter=self.max_iter
        )


class MatchingMNN(MatchingABC):
    """

    """
    def __init__(
            self,
            metric="sqeuclidean",
            k=10,
            use_sparse=True
    ):
        Matching.__init__(self, use_sparse=use_sparse)
        self.metric = metric
        self.k = k

    def _compute_di(self, D, axis):
        """
        Returns the distance of each xi to its kth nearest neighbor
        """
        D_sorted = np.sort(D, axis=axis)
        if axis == 0:
            D_sorted = D_sorted.T
        return D_sorted[:, self.k]

    def _match2(self, x1, x2):
        D = cdist(x1, x2, metric=self.metric)
        dx = self._compute_di(D, axis=1)
        dy = self._compute_di(D, axis=0)
        Dxy = np.minimum.outer(dx, dy)
        return D <= Dxy


class MatchingFusedGW(Matching):
    """

    """
    def __init__(
            self,
            metricM: str = 'euclidean',
            metricM_kwargs: Dict = {},
            metricC1: str= 'euclidean',
            metricC1_kwargs: Dict = {},
            metricC2: str= 'euclidean',
            metricC2_kwargs: Dict = {},
            alpha: float = 0.5,
            loss: str = "square_loss",
            use_sparse: bool = True,
    ):
        Matching.__init__(self, use_sparse=use_sparse)
        self.metricM = metricM
        self.metricM_kwargs = metricM_kwargs
        self.metricC1 = metricC1
        self.metricC1_kwargs = metricC1_kwargs
        self.metricC2 = metricC2
        self.metricC2_kwargs = metricC2_kwargs
        self.alpha = alpha
        self.loss = loss

    def _compute_di(self, x1: np.array, x2: np.array) -> Tuple[np.array]:
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
        C1 = cdist(x1, x1, metric=self.metricC1, **self.metricC1_kwargs)
        C2 = cdist(x2, x2, metric=self.metricC2, **self.metricC2_kwargs)
        return M, C1, C2

    def _match2(self, x1: np.array, x2: np.array) -> np.array:
        """
        Compute optimal transport plan for the FGW problem.
        Parameters
        ----------
        x1: np.array
            A dataset.
        x2: np.array
            A dataset

        Returns
        -------
        T = (xi.shape[0], xj.shape[0]) sparse array, where Tkl is the
        matching strength between xik and xjl.
        """
        n1, n2 = x1.shape[0], x2.shape[0]
        w1, w2 = np.ones(n1) / n1, np.ones(n2) / n2
        M, C1, C2 = self._compute_di(x1, x2)
        C1 /= C1.max()
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
