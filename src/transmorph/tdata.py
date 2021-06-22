#!/usr/bin/env python3

import numpy as np

from .density import sigma_search, normal_kernel_weights
from .utils import tdist

class TData:
    """

    """
    def __init__(self,
                 x: np.ndarray,
                 w: np.ndarray = None,
                 weighted: bool = True,
                 metric: str = "sqeuclidean",
                 normalize: bool = True,
                 scale: float = -1,
                 verbose: bool = True):
        self.x = x.copy()
        self.w = w.copy() if w is not None else None
        self.weighted = weighted
        self.metric = metric
        self.normalize = normalize
        self.scale = scale
        self.verbose = verbose
        self.log("TData initialized, length %i" % len(self))


    def __len__(self):
        return len(self.x)


    def __str__(self):
        return "(TData) of length %i" % len(self)


    def log(self, s: str, end: str = '\n', header: bool = True) -> None:
        # Only prints for now, can later be pipelined into other streams
        if not self.verbose:
            return
        if header:
            print("(Transmorph/TData) > %s" % s, end=end)
        else:
            print(s, end=end)


    def copy(self):
        return TData(
            self.x,
            self.w,
            self.weighted,
            self.metric,
            self.normalize,
            self.scale,
            self.verbose
        )


    def get_weights(self):
        """
        Returns weights associated to self.x points. In charge of computing
        them if needed.
        """
        if self.w is None:
            if not self.weighted:
                self.log("Using uniform weights.")
                self.w = np.ones((len(self),)) / len(self)
            else:
                self.log("Starting the weights selection procedure.")
                dmatrix = tdist(self.x, normalize=self.normalize,
                                metric=self.metric)
                if self.scale == -1:
                    self.log("Searching for sigma...", end=' ')
                    self.scale = sigma_search(dmatrix)
                    self.log("Found: %f" % self.scale, header=False)
                self.log("Solving the QP to find weights...", end=' ')
                self.w = normal_kernel_weights(
                    dmatrix, scale=self.scale
                )
                self.log("Done.", header=False)
            # Trying to correct for approx. errors
            self.w /= np.sum(self.w)
        return self.w

    def get_barycenter(self):
        """

        """
        return (np.diag(self.get_weights()) @ self.x).sum(axis=0)
