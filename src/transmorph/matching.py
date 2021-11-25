#!/usr/bin/env python3

from abc import ABC, abstractmethod
from ot import emd 
from ot.bregman import sinkhorn_stabilized
from ot.gromov import (
    entropic_gromov_wasserstein,
    gromov_wasserstein
)
from scipy.spatial.distance import cdist

import numpy as np


class Matching(ABC):
    """
    A matching is a module containing a function match(x1, ..., xn), able
    to predict matching between dataset samples (possibly fuzzy). 
    """
    def __init__(self):
        self.fitted = False
        self.matchings = []
        

    @abstractmethod
    def _match2(self, x1, x2):
        pass


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
                self.matchings.append(self._match2(di, datasets[j]))
        self.fitted = True
        return self.matchings


class MatchingEMD(Matching):
    """

    """
    def __init__(
            self,
            metric="sqeuclidean",
            metric_kwargs={},
            max_iter=1e6
    ):
        super()
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
            max_iter=1e6
    ):
        super()
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
            max_iter=1e6
    ):
        super()
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
            self.loss,
            max_iter=self.max_iter
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
            max_iter=1e6
    ):
        super()
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
            k=10
    ):
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
