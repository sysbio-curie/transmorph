#!/usr/bin/env python3

import numpy as np

from .density import sigma_search, normal_kernel_weights

class TData:
    """

    """
    def __init__(self,
                 x: np.ndarray,
                 w: np.ndarray = None,
                 weighted: bool = True,
                 scale: float = -1,
                 verbose: bool = True):
        self.x = x.copy()
        self.w = w.copy() if w is not None else None
        self.weighted = weighted
        self.scale = scale
        self.verbose = verbose
        self._print("TData initialized, length %i" % len(self))

    def __len__(self):
        return len(self.x)

    def __str__(self):
        return "(TData) of length %i" % len(self)

    def _print(self, s: str, end: str = '\n', header: bool = True) -> None:
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
            self.scale,
            self.verbose
        )

    def get_weights(self):
        """

        """
        if self.w is None:
            self._print("Computing weights...")
            if not self.weighted:
                self.w = np.ones((len(self),)) / len(self)
            else:
                if self.scale == -1:
                    self._print("Searching for sigma...", end=' ')
                    self.scale = sigma_search(self.x)
                    self._print("Found: %f" % self.scale, header=False)
                self._print("Solving the QP to find weights...")
                self.w = normal_kernel_weights(
                    self.x, scale=self.scale
                )
        return self.w

    def get_barycenter(self):
        """

        """
        return (np.diag(self.get_weights()) @ self.x).sum(axis=0)
