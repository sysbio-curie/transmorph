#!/usr/bin/env python3

import numpy as np
from .integration import ot_integration, ot_transform
from .gw_integration import gw_integration, gw_transform

class Woti:

    def __init__(self,
                 method: str = 'gromov',
                 max_iter: int = 1e6,
                 entropy: bool = False,
                 hreg: float = 1e-3,
                 weighted: bool = True,
                 alpha_qp: float = 1.0,
                 scale: float = 1.0,
                 verbose: bool = True):
        assert method in ('ot', 'gromov'), "Unrecognized method: %s. \
                                            Available methods are 'ot', 'gromov'"
        assert max_iter > 0, "Negative number of iterations."
        assert scale > 0, "Scale must be positive."
        self.method = method
        self.max_iter = max_iter
        self.entropy = entropy
        self.hreg = hreg
        self.weighted = weighted
        self.alpha_qp = alpha_qp
        self.verbose = verbose
        self.scale = scale
        if verbose:
            print("Woti > Successfully initialized.")
            print(str(self))

    def __str__(self):
        return "(Woti) %s based -- max_iter: %i -- %s -- %s" % (
            self.method,
            self.max_iter,
            ( ("entropy regularized, hreg: %f" % self.hreg)
              if self.entropy else "no entropy"),
            ( ("weighted, alpha_qp: %f, scale: %f" % (self.alpha_qp, self.scale))
              if self.weighted else "unweighted")
        )

    def transform(self,
                  xs: np.ndarray,
                  yt: np.ndarray,
                  Mx: np.ndarray = None,
                  My: np.ndarray = None,
                  Mxy: np.ndarray = None) -> np.ndarray:
        if self.method == 'gromov':
            return gw_transform(
                xs, yt, Mx, My, self.max_iter,
                solver=('gw_entropy' if self.entropy else 'gw'),
                hreg=self.hreg, weighted=self.weighted, alpha_qp=self.alpha_qp,
                scale=self.scale, verbose=self.verbose
            )
        if self.method == 'ot':
            return ot_transform(
                xs, yt, Mxy, max_iter=self.max_iter,
                solver=('sinkhorn' if self.entropy else 'emd'),
                hreg=self.hreg, weighted=self.weighted, alpha_qp=self.alpha_qp,
                scale=self.scale, verbose=self.verbose
            )

