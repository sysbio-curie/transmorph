#!/usr/bin/env python3

import numpy as np
import gc

from scipy.sparse import csr_matrix  # Transport plan is usually sparse
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from collections import namedtuple

from .integration import _compute_transport, _transform
from .density import normal_kernel_weights, _get_density, sigma_search
from .tdata import TData
from .utils import col_normalize

# Read-only container for transport results
# TData, TData, array
Transport = namedtuple("Transport", ["src", "tar", "P"])

class Transmorph:
    """
    Optimal transport-related tools for dataset integration.

    This class implements a set of methods related to optimal transport-based
    unsupervised and semi-supervised learning, with the option of using an
    unsupervised heuristic to estimate data points weights prior to optimal
    transport in order to correct for dataset classes unbalance.

    By default, the optimal transport plan will be computed using
    optimal transport metric, but it can be changed to rather use Gromov-
    Wasserstein distance.

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
            integration due to overfitting.

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

    n_comps: int, default = -1
        Number of dimensions to use in the PCA, which is used to compute the cost
        matrix/density. -1 uses the raw dimensionality instead.

    subsampling: float, default = -1
        Fraction of data to use for computing the anchor matching. Set it
        to -1 to use the whole datasets.

    normalize: bool = 1
        Column-normalize matrices before computing cost matrix.

    verbose: int, default = 1
        Defines the logging level.
        0: Disabled
        1: Informations
        2: Debug
    """

    def __init__(self,
                 method: str = 'ot',
                 max_iter: int = 1e6,
                 entropy: bool = False,
                 hreg: float = 1e-3,
                 weighted: bool = True,
                 alpha_qp: float = 1.0,
                 scale: float = -1,
                 metric: str = 'sqeuclidean',
                 n_comps: int = -1,
                 subsampling: float = -1,
                 normalize: bool = True,
                 verbose: bool = False):

        assert method in ('ot', 'gromov'), "Unrecognized method: %s. \
                                            Available methods are 'ot', 'gromov'"
        assert max_iter > 0, "Negative number of iterations."
        assert scale > 0 or scale == -1, "Scale must be positive."
        assert subsampling == -1 or 0 < subsampling <= 1, "Invalid subsampling ratio."

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
        self.n_comps = n_comps
        self.subsampling = subsampling
        self.normalize = normalize
        # Cache for transport plans
        self.tdata_x = None
        self.tdata_y = None
        self.transports = []
        self._log("Successfully initialized.")
        self._log(str(self), level=2)

    def __str__(self) -> str:
        return f'<Transmorph> {self.method}-based.\n\
        -- fitted: {self.fitted}\n\
        -- max_iter: {self.max_iter}\n\
        -- entropy: {self.entropy}\n\
        -- hreg: {self.hreg}\n\
        -- weighted: {self.weighted}\n\
        -- metric: {self.metric}\n\
        -- n_comps: {self.n_comps}\n\
        -- subsampling: {self.subsampling}\n\
        -- normalize: {self.normalize}\n\
        -- n_integrations: {len(self.transports)}'

    def _log(self, s: str, end: str = '\n', header: bool = True, level=2) -> None:
        # Only prints for now, can later be pipelined into other streams
        if level > self.verbose:
            return
        if header:
            s = "(Transmorph) > %s" % s
        print(s, end=end)

    def fit(self,
            xs: np.ndarray,
            yt: np.ndarray,
            Mx: np.ndarray = None,
            My: np.ndarray = None,
            Mxy: np.ndarray = None) -> None:
        """
        Computes the optimal transport plan between two empirical distributions,
        with parameters specified during initialization. Caches the result
        in the Transmorph object.

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
        ### Computing weights
        n, m = len(xs), len(yt)
        assert n > 0, "Empty source matrix."
        assert m > 0, "Empty reference matrix."

        self.fitted = False
        self.transports = [] # Clearing for now

        xs_nrm, yt_nrm = None, None
        if self.normalize:
            self._log("Computing normalized view...")
            xs_nrm, yt_nrm = col_normalize(xs), col_normalize(yt)

        xs_red, yt_red = None, None
        if self.n_comps != -1:
            self._log("Computing PC view, %i PCs..." % self.n_comps)
            pca = PCA(n_components=self.n_comps)
            xy = None
            if self.normalize:
                xy = np.concatenate( (xs_nrm, yt_nrm), axis=0 )
            else:
                xy = np.concatenate( (xs, yt), axis=0 )
            xy = pca.fit_transform(xy)
            xs_red, yt_red = xy[:n], xy[n:]

        # Full TDatas
        self.tdata_x = TData(xs,
                             x_nrm=xs_nrm,
                             x_red=xs_red,
                             weighted=self.weighted,
                             metric=self.metric,
                             scale=self.scale,
                             alpha_qp=self.alpha_qp,
                             verbose=self.verbose)

        self.tdata_y = TData(yt,
                             x_nrm=yt_nrm,
                             x_red=yt_red,
                             weighted=self.weighted,
                             metric=self.metric,
                             scale=self.scale,
                             alpha_qp=self.alpha_qp,
                             verbose=self.verbose)

        # Computing cost matrices if necessary.
        if self.method == 'ot' and Mxy is None:
            self._log("Using metric %s as a cost for Mxy. Normalization: %r" %
                     (self.metric, self.normalize))
            assert xs.shape[1] == yt.shape[1], (
                "Dimension mismatch (%i != %i)" % (xs.shape[1], yt.shape[1]))
            Mxy = self.tdata_x.distance(self.tdata_y)
        if self.method == 'gromov' and Mx is None:
            self._log("Using metric %s as a cost for Mx. Normalization: %r" %
                     (self.metric, self.normalize))
            Mx = self.tdata_x.distance()
        if self.method == 'gromov' and My is None:
            self._log("Using metric %s as a cost for My. Normalization: %r" %
                     (self.metric, self.normalize))
            My = self.tdata_y.distance()

        # Partial cost matrices
        Mxyi, Mxi, Myi = Mxy, Mx, My
        xsi_raw, xsi_nrm, xsi_red = xs, xs_nrm, xs_red
        yti_raw, yti_nrm, yti_red = yt, yt_nrm, yt_red
        iter_max = 1 if self.subsampling == -1 else (1 + int(1/self.subsampling))**2

        # Main subsampling loop
        for n_iter in range(iter_max):

            sel_x, sel_y = np.ones(n).astype(bool), np.ones(m).astype(bool)
            if self.subsampling != -1:
                self._log(f'Subsampling, iteration {n_iter+1}/{iter_max}')
                sel_x = np.random.rand(n) < self.subsampling
                sel_y = np.random.rand(m) < self.subsampling

            tdata_x = TData(xs[sel_x],
                            x_nrm=(None if xs_nrm is None else xs_nrm[sel_x]),
                            x_red=(None if xs_red is None else xs_red[sel_x]),
                            weighted=self.weighted,
                            metric=self.metric,
                            scale=self.scale,
                            alpha_qp=self.alpha_qp,
                            verbose=self.verbose)

            tdata_y = TData(yt[sel_y],
                            x_nrm=(None if yt_nrm is None else yt_nrm[sel_y]),
                            x_red=(None if yt_red is None else yt_red[sel_y]),
                            weighted=self.weighted,
                            metric=self.metric,
                            scale=self.scale,
                            alpha_qp=self.alpha_qp,
                            verbose=self.verbose)

            # Computing cost matrices if necessary.
            # Copying for C-contiguity, required by pot
            if self.method == 'ot':
                Mxyi = Mxy[sel_x][:,sel_y].copy()
            if self.method == 'gromov':
                Mxi = Mx[sel_x][:,sel_x].copy()
                Myi = My[sel_y][:,sel_y].copy()

            # Projecting source to ref
            self._log("Computing transport plan (%s)..." % self.method)
            Pxy = _compute_transport(
                tdata_x.weights(), tdata_y.weights(),
                method=self.method, Mxy=Mxyi, Mx=Mxi, My=Myi,
                max_iter=self.max_iter, entropy=self.entropy,
                hreg=self.hreg)

            self.transports.append(Transport(tdata_x, tdata_y, Pxy))

        self.fitted = True
        self._log("Transmorph fitted.")


    def transform(self,
                  xs: np.ndarray = None,
                  jitter: bool = True,
                  jitter_std: float = .01,
                  neighbors_smoothing: int = 1) -> np.ndarray:
        """
        Applies optimal transport integration. Transmorph must be fitted beforehand.

        xs: np.ndarray, default = None
            Source dataset to transform.

        jitter: bool, default = True
            Adds a little bit of random scattering to the final results. Helps
            downstream methods such as UMAP to converge in some cases.

        jitter_std: float, default = 0.01
            Jittering standard deviation.

        Returns:
        --------
        (n,d1) np.ndarray, of xs integrated onto yt.
        """
        assert self.fitted, "Transmorph must be fitted first."
        self._log("Projecting dataset...")
        xt = _transform(self.tdata_x,
                        self.transports,
                        jitter=jitter,
                        jitter_std=jitter_std,
                        n_neighbors=neighbors_smoothing)
        self._log("Terminated")
        return xt


    def fit_transform(self,
                      xs: np.ndarray,
                      yt: np.ndarray,
                      Mx: np.ndarray = None,
                      My: np.ndarray = None,
                      Mxy: np.ndarray = None,
                      jitter: bool = True,
                      jitter_std: float = .01,
                      neighbors_smoothing: int = 1) -> np.ndarray:
        """
        Shortcut, fit() -> transform()
        """
        self.fit(xs, yt, Mx, My, Mxy)
        return self.transform(xs,
                              jitter=jitter,
                              jitter_std=jitter_std,
                              neighbors_smoothing=neighbors_smoothing)


    def label_transfer(self, y_labels: np.ndarray) -> np.ndarray:
        """
        Uses the optimal tranport plan to infer xs labels based
        on yt ones in a semi-supervised fashion.

        Parameters:
        -----------
        y_labels: (m,) np.ndarray
            Labels of reference dataset points.

        Returns:
        --------
        (n,) np.ndarray, predicted labels for source dataset points.
        """
        assert self.fitted, "Transmorph must be fitted first."
        assert len(transports) == 1, "Subsampling unsupported yet."
        return y_labels[np.argmax(self.transports[0]["P"].toarray(), axis=1)]
