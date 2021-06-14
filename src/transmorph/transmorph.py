#!/usr/bin/env python3

import numpy as np
from .integration import _compute_transport, _transform
from .density import normal_kernel_weights, _get_density, sigma_search

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
        # Cache for dataset weights
        self.wx = None
        self.wy = None
        # Cache datasets to avoid weights recomputation
        self.xs = None # Source dataset
        self.yt = None # Reference dataset
        self._print("Successfully initialized.\n%s" % str(self))

    def copy(self):
        c = Transmorph(self.method, self.max_iter, self.entropy, self.hreg,
                       self.weighted, self.alpha_qp, self.scale, self.metric,
                       self.verbose)
        c.fitted = self.fitted
        c.transport_plan = self.transport_plan.copy()
        c.wx = self.wx
        c.wy = self.wy
        c.xs = self.xs
        c.yt = self.yt
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

    def sigma_search(self, xs):
        """
        Returns the RBF sigma maximizing density entropy.
        """
        self._print("Sigma search...", end=' ')
        sigma = sigma_search(xs)
        self._print("Found: %f" % sigma, header=False)
        return sigma

    def compute_weights(self, x: np.ndarray) -> np.ndarray:
        """
        Returns weights for points in x so that kernel density
        is uniform over the dataset.

        Returns:
        --------
        (n,) vector of positive real numbers that sums to 1.

        Parameters:
        -----------
        x: np.ndarray (n,d)
            Input dataset
        """
        assert len(x) > 0, "Empty array."

        # Uniform case
        if not self.weighted:
            n = len(x)
            return np.array([1 / n] * n)

        # Search the cache
        if x is self.xs:
            self._print("Using cached values (wx).")
            return self.wx
        if x is self.yt:
            self._print("Using cached values (wy).")
            return self.wy

        # Compute weights
        if self.scale == -1:
            sigma = self.sigma_search(x)
        else:
            sigma = self.scale
        self._print("Solving the QP...", end=' ')
        w = normal_kernel_weights(
            x, alpha_qp=self.alpha_qp, scale=sigma
        )
        self._print("Done.", header=False)
        return w

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

        # Computing weights if necessary
        self._print("Computing source weights...")
        self.wx = self.compute_weights(xs)
        self._print("Computing target weights...")
        self.wy = self.compute_weights(yt)
        self.xs = xs
        self.yt = yt

        # Projecting source to ref
        self._print("Computing transport plan (%s)..." % self.method)
        self.transport_plan = _compute_transport(
            xs, yt, self.wx, self.wy, method=self.method, metric=self.metric,
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
        m, d = self.yt.shape
        n, mP = self.transport_plan.shape
        nw = len(self.wx)
        assert m == mP, "Inconsistent dimension between reference and transport."
        assert n == nw, "Inconsistent dimension between weights and transport."
        self._print("Projecting dataset...")
        return _transform(self.wx, self.yt, self.transport_plan,
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

    def euclidean_barycenter(self):
        assert self.fitted, "Transmorph must be fitted first."
        return (
            (np.diag(self.wx) @ self.xs).sum(axis=0),
            (np.diag(self.wy) @ self.yt).sum(axis=0)
        )
