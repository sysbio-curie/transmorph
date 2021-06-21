#!/usr/bin/env python3

import numpy as np
from scipy.sparse import csr_matrix

from .integration import _compute_transport, _transform
from .density import normal_kernel_weights, _get_density, sigma_search
from .tdata import TData

class Transmorph:
    """
    Optimal transport-related tools for dataset integration.

    This class implements a set of methods related to optimal transport-based
    unsupervised and semi-supervised learning, with the option of using an
    unsupervised heuristic to estimate data points weights prior to optimal
    transport in order to correct for dataset classes unbalance.

    By default, the optimal transport plan will be computed using
    Gromov-Wasserstein metric, but it can be changed to rather use optimal
    transport (Wasserstein distance).

    Parameters:
    -----------
    method: str in ('ot', 'gromov'), default = 'ot'
        Method to use in order to compute optimal transport plan.
        OT stands for optimal transport, and requires to define a metric
            between points in different distributions. It is in general
            more reliable in the case of datasets presenting symmetries.
        Gromov stands for Gromov-Wasserstein (GW), and only requires the metric
            in both spaces. It is in this sense more general, but is
            invariant to dataset isometries which can lead to poor
            integration.

    max_iter: int, default = 1e6
        Maximum number of iterations for OT/GW.

    entropy: bool, default = False
        Enables the entropy regularization for OT/GW problem, solving it
        approximately but more efficiently using Sinkhorn's algorithm
        (Cuturi 2013)

    hreg: float, default = 1e-3
        Entropic regularization parameter. The lower the more accurate the result
        will be, at a cost of extra computation time.

    weighted: bool, default = True
        Enables the weights correction to deal with unbalanced datasets. It requires
        to solve a QP of dimensionality equal to dataset size, so it does not scale
        for now above 10^4 data points.

    alpha_qp: float, default = 1.0
        Parameter for the QP solver (osqp)

    scale: float, default = -1
        Standard deviation of Gaussian RBFs used in the weights selection. May need
        some tuning. Set it to -1 to use an adaptive sigma.

    metric: str (see scipy.spatial.distance.cdist)
        Default metric to use.

    verbose: bool, default = False
        Enable logging.
    """

    def __init__(self,
                 method: str = 'ot',
                 max_iter: int = 1e6,
                 entropy: bool = False,
                 hreg: float = 1e-3,
                 weighted: bool = True,
                 alpha_qp: float = 1.0,
                 scale: float = -1,
                 metric: str = 'euclidean',
                 verbose: bool = False):
        assert method in ('ot', 'gromov'), "Unrecognized method: %s. \
                                            Available methods are 'ot', 'gromov'"
        assert max_iter > 0, "Negative number of iterations."
        assert scale > 0 or scale == -1, "Scale must be positive."
        self.fitted = False
        self.method = method
        self.max_iter = max_iter
        self.entropy = entropy
        self.hreg = hreg
        self.weighted = weighted
        self.alpha_qp = alpha_qp
        self.verbose = verbose
        self.scale = scale
        self.metric = metric
        # Cache for transport plan
        self.transport_plan = None
        # Cache for datasets (TData)
        self.xs = None
        self.yt = None
        self._print("Successfully initialized.\n%s" % str(self))

    def copy(self):
        c = Transmorph(
            self.method, self.max_iter, self.entropy, self.hreg,
            self.weighted, self.alpha_qp, self.scale, self.metric,
            self.verbose)
        c.fitted = self.fitted
        c.transport_plan = self.transport_plan.copy()
        c.xs = self.xs.copy()
        c.yt = self.yt.copy()
        return c


    def __str__(self) -> str:
        return "(Transmorph) %s based \n \
                -- max_iter: %i \n \
                -- %s \n \
                -- %s \n \
                -- metric: %s" % (
            self.method,
            self.max_iter,
            ( ("entropy regularized, hreg: %f" % self.hreg)
              if self.entropy else "no entropy"
            ),
            ( ("weighted, alpha_qp: %f, scale: %s" % (
                self.alpha_qp,
                str(self.scale) if self.scale != -1 else 'adaptive'))
              if self.weighted else "unweighted"
            ),
            self.metric
        )

    def _print(self, s: str, end: str = '\n', header: bool = True) -> None:
        # Only prints for now, can later be pipelined into other streams
        if not self.verbose:
            return
        if header:
            print("(Transmorph) > %s" % s, end=end)
        else:
            print(s, end=end)

    def fit(self,
            xs: np.ndarray,
            yt: np.ndarray,
            Mx: np.ndarray = None,
            My: np.ndarray = None,
            Mxy: np.ndarray = None) -> None:
        """
        Computes the optimal transport plan between two empirical distributions,
        with parameters specified during initialization.


        Parameters:
        -----------
        xs: (n,d0) np.ndarray
            Source data points cloud.

        yt: (m,d1) np.ndarray
            Target data points cloud.

        Mx: (n,n) np.ndarray
            Pairwise metric matrix for xs. Only relevant for GW. If
            None, Euclidean distance is used.

        My: (m,m) np.ndarray
            Pairwise metric matrix for yt. Only relevant for GW. If
            None, Euclidean distance is used.

        Mxy: (n,m) np.ndarray
            Pairwise metric matrix between xs and yt. Only relevant for OT.
            If None, Euclidean distance is used.
        """
        # Computing weights
        n, m = len(xs), len(yt)
        assert n > 0, "Empty source matrix."
        assert m > 0, "Empty reference matrix."

        self.fitted = False

        # Creating TData objects if necessary
        if self.xs is None or not np.array_equal(self.xs.x, xs):
            self.xs = TData(xs, None, self.weighted, self.scale, self.verbose)

        if self.yt is None or not np.array_equal(self.yt.x, yt):
            self.yt = TData(yt, None, self.weighted, self.scale, self.verbose)

        # Projecting source to ref
        self._print("Computing transport plan (%s)..." % self.method)
        self.transport_plan = _compute_transport(
            xs, yt, self.xs.get_weights(), self.yt.get_weights(),
            method=self.method, metric=self.metric,
            Mxy=Mxy, Mx=Mx, My=My,
            max_iter=self.max_iter, entropy=self.entropy,
            hreg=self.hreg)

        self.fitted = True
        self._print("Transmorph fitted.")


    def transform(self, jitter: bool = True, jitter_std: float = .01) -> np.ndarray:
        """
        Applies optimal transport integration. Transmorph must be fitted beforehand.

        jitter: bool, default = True
            Adds a little bit of random scattering to the final results. Helps
            downstream methods such as UMAP in some cases.

        jitter_std: float, default = 0.01
            Jittering standard deviation.

        Returns:
        --------
        (n,d1) np.ndarray, of xs integrated onto yt.
        """
        assert self.fitted, "Transmorph must be fitted first."
        wy = self.yt.get_weights()
        yt = self.yt.x
        m, d = yt.shape
        n, mP = self.transport_plan.shape
        mw = len(wx)
        assert m == mP, "Inconsistent dimension between reference and transport."
        assert mP == mw, "Inconsistent dimension between weights and transport."
        self._print("Projecting dataset...")
        return _transform(wy, yt, self.transport_plan,
                          jitter=jitter, jitter_std=jitter_std)


    def fit_transform(self,
                      xs: np.ndarray,
                      yt: np.ndarray,
                      Mx: np.ndarray = None,
                      My: np.ndarray = None,
                      Mxy: np.ndarray = None,
                      jitter: bool = True,
                      jitter_std: float = .01) -> np.ndarray:
        """
        Shortcut, fit() -> transform()
        """
        self.fit(xs, yt, Mx, My, Mxy)
        return self.transform(jitter=jitter, jitter_std=jitter_std)


    def label_transfer(self, y_labels: np.ndarray) -> np.ndarray:
        """
        Uses the optimal tranport plan to infer xs labels based
        on yt ones in a semi-supervised fashion.
        """
        assert self.fitted, "Transmorph must be fitted first."
        return y_labels[np.argmax(self.transport_plan.toarray(), axis=1)]

