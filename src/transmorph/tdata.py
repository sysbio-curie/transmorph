#!/usr/bin/env python3

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

from .density import sigma_search, normal_kernel_weights

class TData:
    """

    """
    def __init__(self,
                 x: np.ndarray,
                 w: np.ndarray = None,
                 weighted: bool = True,
                 metric: str = "sqeuclidean",
                 n_comps: int = -1,
                 normalize: bool = True,
                 scale: float = -1,
                 verbose: bool = True):

        self.x = x.copy()
        self.x_nrm = self.x # Normalized view
        self.x_red = self.x # PC view
        self.dmatrix = None
        self.w = w.copy() if w is not None else None
        self.weighted = weighted
        self.metric = metric
        self.n_comps = n_comps
        self.normalize = normalize
        self.scale = scale
        self.verbose = verbose

        # Columns normalization
        if self.normalize:
            x_std = x.std(axis=0)
            x_std[x_std == 0] = 1
            self.x_nrm = x / x_std

        if self.n_comps != -1 and self.weighted:
            self._log("Computing a %i components PCA..." % n_comps, end=' ')
            pca = PCA(n_components=n_comps)
            self.x_red = pca.fit_transform(self.x_nrm)
            self._log("Done.", header=False)

        self._log("TData initialized, length %i" % len(self))


    def __len__(self):
        return len(self.x)


    def __str__(self):
        return "<TData> of length %i" % len(self)


    def _log(self, s: str, end: str = '\n', header: bool = True, level=1) -> None:
        # Only prints for now, can later be pipelined into other streams
        if level > self.verbose:
            return
        if header:
            s = "(Transmorph/TData) > %s" % s
        print(s, end=end)


    def copy(self):
        t = TData(
            self.x,
            self.w,
            self.weighted,
            self.metric,
            self.n_comps,
            self.normalize,
            self.scale,
            self.verbose
        )
        t.x_nrm = self.x_nrm.copy()
        t.x_red = self.x_red.copy()
        if self.dmatrix is not None:
            t.dmatrix = self.dmatrix.copy()

        return t

    def get_dmatrix(self):
        """
        Reuturns the (n,n) pairwise distance matrix
        """
        if self.dmatrix is None:
            self.dmatrix = cdist(self.x_red, self.x_red, metric=self.metric)
        return self.dmatrix


    def get_weights(self):
        """
        Returns weights associated to self.x points. In charge of computing
        them if needed.
        """
        if self.w is None:
            if not self.weighted:
                self._log("Using uniform weights.")
                self.w = np.ones((len(self),)) / len(self)
            else:
                self._log("Starting the weights selection procedure.")
                if self.scale == -1:
                    self._log("Searching for sigma...", end=' ')
                    self.scale = sigma_search(self.get_dmatrix())
                    self._log("Found: %f" % self.scale, header=False)
                self._log("Solving the QP to find weights...", end=' ')
                self.w = normal_kernel_weights(self.get_dmatrix(), scale=self.scale)
                self._log("Done.", header=False)
        return self.w

    def get_barycenter(self):
        """

        """
        return (np.diag(self.get_weights()) @ self.x).sum(axis=0)
