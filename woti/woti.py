#!/usr/bin/env python3

import numpy as np
from .integration import _compute_transport, _transform
from .density import normal_kernel_weights

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
        # Cache
        self.transport_plan = None
        self.wx = None
        self.yt = None
        if verbose:
            self.print("Successfully initialized.\n%s" % str(self))


    def __str__(self):
        return "(Woti) %s based -- max_iter: %i -- %s -- %s" % (
            self.method,
            self.max_iter,
            ( ("entropy regularized, hreg: %f" % self.hreg)
              if self.entropy else "no entropy"),
            ( ("weighted, alpha_qp: %f, scale: %f" % (self.alpha_qp, self.scale))
              if self.weighted else "unweighted")
        )

    def print(self, s):
        print("WOTi > %s" % s)

    def is_fitted(self):
        return not (
            self.transport_plan is None
            or self.wx is None
            or self.yt is None)

    def fit(self,
            xs: np.ndarray,
            yt: np.ndarray,
            Mx: np.ndarray = None,
            My: np.ndarray = None,
            Mxy: np.ndarray = None) -> np.ndarray:

        # Computing weights
        n, m = len(xs), len(yt)
        assert n > 0, "Empty source matrix."
        assert m > 0, "Empty reference matrix."
        self.yt = yt
        if not self.weighted:
            self.wx, wy = np.array([1 / n] * n), np.array([1 / m] * m)
        else:
            if self.verbose:
                self.print("Computing source distribution weights...")
            self.wx = normal_kernel_weights(
                xs, alpha_qp=self.alpha_qp, scale=self.scale
            )
            if self.verbose:
                self.print("Computing reference distribution weights...")
            wy = normal_kernel_weights(
                yt, alpha_qp=self.alpha_qp, scale=self.scale
            )

        # Correcting approximation error
        self.wx /= np.sum(self.wx)
        wy /= np.sum(wy)

        if self.verbose:
            if self.method == "ot":
                self.print("Computing optimal transport plan...")
            if self.method == "gromov":
                self.print("Computing Gromov-Wasserstein plan...")

            self.transport_plan = _compute_transport(
                xs, yt, self.wx, wy, method=self.method, Mxy=Mxy, Mx=Mx, My=My,
                max_iter=self.max_iter, entropy=self.entropy,
                hreg=self.hreg, verbose=self.verbose)

        self.print("WOTi fitted.")


    def transform(self) -> np.ndarray:
        assert self.is_fitted(), "WOTi must be fitted first."
        m, d = self.yt.shape
        n, mP = self.transport_plan.shape
        nw = len(self.wx)
        assert m == mP, "Inconsistent dimension between reference and transport."
        assert n == nw, "Inconsistent dimension between weights and transport."
        self.print("Projecting dataset...")
        return _transform(self.wx, self.yt, self.transport_plan)


    def fit_transform(self,
            xs: np.ndarray,
            yt: np.ndarray,
            Mx: np.ndarray = None,
            My: np.ndarray = None,
            Mxy: np.ndarray = None) -> np.ndarray:
        self.fit(xs, yt, Mx, My, Mxy)
        return self.transform()


    def label_transfer(
            xs: np.ndarray,
            yt: np.ndarray,
            labels: np.ndarray,
            Mx: np.ndarray = None,
            My: np.ndarray = None,
            Mxy: np.ndarray = None) -> np.ndarray:
        return
