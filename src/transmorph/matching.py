#!/usr/bin/env python3

from abc import ABC, abstractmethod
from ot import emd 
from ot.bregman import sinkhorn_stabilized
from ot.gromov import (
    entropic_gromov_wasserstein,
    gromov_wasserstein
)
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix

import numpy as np


class Matching(ABC):
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
    def __init__(self, use_sparse=True):
        self.fitted = False
        self.matchings = []
        self.use_sparse = use_sparse    
        

    @abstractmethod
    def _match2(self, x1, x2):
        pass


    def get(self, i, j, normalize=False):
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
        T = (xi.shape[0], xj.shape[0]) sparse array, where Tkl is the
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


    def match(self, *datasets):
        """
        Matches all pairs of different datasets together. Returns results
        in a dictionary, where d[i,j] is the matching between datasets i
        and j represented as a (ni, nj) numpy array -- possibly fuzzy.

        Parameters:
        -----------
        *datasets: list of datasets
            List of at least two datasets. 
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


class MatchingEMD(Matching):
    """

    """
    def __init__(
            self,
            metric="sqeuclidean",
            metric_kwargs={},
            max_iter=1e6,
            use_sparse=True
    ):
        Matching.__init__(self, use_sparse=use_sparse)
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


class MatchingSinkhorn(Matching):
    """

    """
    def __init__(
            self,
            metric="sqeuclidean",
            metric_kwargs={},
            epsilon=1e-2,
            max_iter=5e2,
            use_sparse=True
    ):
        Matching.__init__(self, use_sparse=use_sparse)
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


class MatchingGW(Matching):
    """

    """
    def __init__(
            self,
            metric="sqeuclidean",
            metric_kwargs={},
            metric2=None,
            metric2_kwargs={},
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


class MatchingGWEntropic(Matching):
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


class MatchingMNN(Matching):
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
