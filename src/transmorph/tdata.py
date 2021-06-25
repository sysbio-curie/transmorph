#!/usr/bin/env python3

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

from .density import sigma_search, normal_kernel_weights
from .utils import col_normalize

class TData:
    """
    # TODO: write doctring
    """
    def __init__(self,
                 x_raw: np.ndarray,
                 x_nrm: np.ndarray = None,
                 x_red: np.ndarray = None,
                 weighted: bool = True,
                 metric: str = "sqeuclidean",
                 scale: float = -1,
                 alpha_qp: float = 1.0,
                 verbose: bool = 0):

        # Keeping precomp. ArrayViews to save space. Must be used with care.
        # Usage priority in methods: x_red > x_nrm > x_raw
        self.x_raw = x_raw  # Raw matrix                (n, D)
        self.x_nrm = x_nrm  # Feature-normalized matrix (n, D)
        self.x_red = x_red  # Dimension-reduced matrix  (n, d)

        self._weightseighted = weighted
        self.metric = metric
        self.scale = scale
        self.alpha_qp = alpha_qp
        self.verbose = verbose

        # Will be computed on the fly if needed
        self.dmatrix = None
        self._weights = None

        self._log("TData initialized, length %i" % len(self), level=3)


    def __len__(self):
        return len(self.x_raw)


    def __str__(self):
        return "<TData> of length %i" % len(self)


    def _log(self, s: str, end: str = '\n', header: bool = True, level=2) -> None:
        # Only prints for now, can later be pipelined into other streams
        if level > self.verbose:
            return
        if header:
            s = "(Transmorph/TData) > %s" % s
        print(s, end=end)


    def distance(self, other=None):
        """
        Reuturns the inner pairwise distance matrix by default, or the
        (n,m) pairwise distance matrix if another TData is provided.

        Priority is red > nrm > raw
        """
        # Cacheing the inner pairwise matrix if needed
        if other is None and self.dmatrix is None:
            if self.x_red is not None:
                self.dmatrix = cdist(self.x_red, self.x_red, metric=self.metric)
            elif self.x_nrm is not None:
                self.dmatrix = cdist(self.x_nrm, self.x_nrm, metric=self.metric)
            else:
                self.dmatrix = cdist(self.x_raw, self.x_raw, metric=self.metric)

        if other is None:
            dmatrix = self.dmatrix
        else:
            x, y = None, None
            if (self.x_red is not None
                and other.x_red is not None
                and self.x_red.shape[1] == other.x_red.shape[1]):
                x, y = self.x_red, other.x_red
            elif (self.x_nrm is not None
                  and other.x_nrm is not None
                  and self.x_nrm.shape[1] == other.x_nrm.shape[1]):
                x, y = self.x_nrm, other.x_nrm
            elif (self.x_raw is not None
                  and other.x_raw is not None
                  and self.x_raw.shape[1] == other.x_raw.shape[1]):
                x, y = self.x_raw, other.x_raw
            else:
                raise ValueError("Incompatible shapes in all views.")

            dmatrix = cdist(x, y, metric=self.metric)

        return dmatrix


    def weights(self):
        """
        Returns weights associated to self.x points. In charge of computing
        them if needed.
        """
        if self._weights is None:
            if not self.weighted:
                self._log("Using uniform weights.")
                self._weights = np.ones((len(self),)) / len(self)
            else:
                self._log("Starting the weights selection procedure.")
                if self.scale == -1:
                    self._log("Searching for sigma...", end=' ')
                    self.scale = sigma_search(self.distance())
                    self._log("Found: %f" % self.scale, header=False)
                self._log("Solving the QP to find weights...", end=' ')
                self._weights = normal_kernel_weights(self.distance(),
                                                      scale=self.scale,
                                                      alpha_qp=self.alpha_qp)
                self._log("Done.", header=False)
        return self._weights


    def get_barycenter(self):
        """

        """
        return (np.diag(self._weightseights()) @ self.x_raw).sum(axis=0)
