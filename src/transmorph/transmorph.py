#!/usr/bin/env python3

import numpy as np

from sklearn.decomposition import PCA
from sklearn.utils import check_array
from scipy.spatial.distance import cdist
from collections import namedtuple

from .integration import compute_transport
from .integration import transform
from .tdata import TData
from .utils import col_normalize

# TODO: Transport -> separate class
# Read-only container for transport results
# TData, TData, array
Transport = namedtuple("Transport", ["src", "tar", "P"])

def penalize_per_label(M, xs_labels, yt_labels):
    return M + M.max()*(xs_labels[:,None] != yt_labels)

def weight_per_label(xs_labels, yt_labels):
    # Weighting by cell type proportion

    n, m = len(xs_labels), len(yt_labels)
    all_labels = list(set(xs_labels).union(set(yt_labels)))

    # Labels specific to xs/yt
    labels_specx = [ i for (i, li) in enumerate(all_labels)
                     if li not in yt_labels ] 
    labels_specy = [ i for (i, li) in enumerate(all_labels)
                     if li not in xs_labels ] 
    labels_common = [ i for (i, li) in enumerate(all_labels)
                      if i not in labels_specx
                      and i not in labels_specy ]

    # Fequency of each label
    xs_freqs = np.array([
        np.sum(xs_labels == li) / n for li in all_labels
    ])
    yt_freqs = np.array([
        np.sum(yt_labels == li) / m for li in all_labels
    ])

    # Only accounting for common labels
    norm_x, norm_y = (
        np.sum(xs_freqs[labels_common]),
        np.sum(yt_freqs[labels_common])
    )
    rel_freqs = np.zeros(len(all_labels))
    rel_freqs[labels_common] = (
        yt_freqs[labels_common] * norm_x / (xs_freqs[labels_common] * norm_y)
    )
    
    # Correcting weights with respect to label frequency
    wx, wy = np.ones(n) / n, np.ones(m) / m
    for fi, li in zip(rel_freqs, all_labels):
        wx[xs_labels == li] *= fi
    for i in labels_specx + labels_specy:
        wy[yt_labels == all_labels[i]] = 0

    wx, wy = wx / wx.sum(), wy / wy.sum()
    return wx, wy

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
                 unbalanced: bool = False,
                 max_iter: int = 1e6,
                 entropy: bool = False,
                 hreg: float = 1e-3,
                 mreg: float = 1e-3,
                 weighted: bool = True,
                 alpha_qp: float = 1.0,
                 scale: float = -1,
                 metric: str = 'sqeuclidean',
                 n_comps: int = -1,
                 subsampling: float = -1,
                 normalize: bool = True,
                 verbose: bool = False):

        self.method = method
        self.unbalanced = unbalanced
        self.max_iter = int(max_iter)
        self.entropy = entropy
        self.hreg = hreg
        self.mreg = mreg
        self.weighted = weighted
        self.alpha_qp = alpha_qp
        self.scale = scale
        self.metric = metric
        self.n_comps = n_comps
        self.subsampling = subsampling
        self.normalize = normalize
        self.verbose = verbose

        self.validate_parameters()

        # Cache for transport plans
        self.tdata_x = None
        self.tdata_y = None
        self.transports = []
        self.fitted = False

        self._log("Successfully initialized.")
        self._log(str(self), level=2)

    def validate_parameters(self):


        assert self.method in ('ot', 'gromov'), \
            "Unrecognized method: %s. Available methods \
            are 'ot', 'gromov'" % self.method

        assert self.max_iter > 0, \
            "Negative number of iterations."

        assert self.scale > 0 or self.scale == -1, \
            "Scale must be positive."

        assert self.subsampling == -1 or 0 < self.subsampling <= 1, \
            "Invalid subsampling ratio %f." % self.subsampling

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
            xs_labels: np.ndarray = None,
            yt_labels: np.ndarray = None,
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

        xs_labels: (n,) np.ndarray
            Labels of xs points, for the semi-supervised integration.

        yt_labels: (m,) np.ndarray
            Label of yt points, for the semi-supervised integration.

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
        ### Verifying parameters

        # Data matrices
        n, m = len(xs), len(yt)
        assert n > 0, "Empty source matrix."
        assert m > 0, "Empty reference matrix."

        xs = check_array(xs, dtype=np.float32, order="C")
        yt = check_array(yt, dtype=np.float32, order="C")

        # Use sample labels frequency to estimate weights?
        is_label_weighted = (
            self.weighted
            and xs_labels is not None
            and yt_labels is not None
        )

        # Verifying/formatting labels
        # Then, computing relative weights
        if is_label_weighted:
            assert xs_labels.shape[0] == n, \
                "Inconsistent dimension for labels in source dataset, \
                %i != %i" % (xs_labels.shape[0], n)
            assert yt_labels.shape[0] == m, \
                "Inconsistent dimension for labels in reference dataset, \
                %i != %i" % (yt_labels.shape[0], m)

        # Cost matrices
        if self.method == 'ot' and Mxy is not None:
            Mxy = check_array(Mxy, dtype=np.float32, order="C")
            assert Mxy.shape == (n, m), \
                "Inconsistent dimension between user-provided cost \
                matrix and datasets size. Expected: (%i,%i), found: (%i,%i)" \
                % (n, m, *Mxy.shape)
        if self.method == 'gromov' and Mx is not None:
            Mx = check_array(Mx, dtype=np.float32, order="C")
            assert Mx.shape == (n, n), \
                "Inconsistent dimension between user-provided cost \
                matrix and source dataset size. \
                Expected: (%i,%i), found: (%i,%i)" \
                % (n, n, *Mx.shape)
        if self.method == 'gromov' and My is not None:
            My = check_array(My, dtype=np.float32, order="C")
            assert My.shape == (m, m), \
                "Inconsistent dimension between user-provided cost \
                matrix and reference dataset size. \
                Expected: (%i,%i), found: (%i,%i)" \
                % (m, m, *My.shape)

        ### Starting the procedure
        self.fitted = False
        self.transports = [] # Clearing for now

        # STD-normalization
        xs_nrm, yt_nrm = None, None
        if self.normalize:
            self._log("Computing normalized view...")
            xs_nrm, yt_nrm = col_normalize(xs), col_normalize(yt)

        # Dimensionality reduction, only needed if asked and
        # no cost matrix is provided.
        xs_red, yt_red = None, None
        if self.n_comps != -1:
            if self.method == 'ot' and Mxy is None:
                self._log("Computing joint PC view, %i PCs..."\
                          % self.n_comps)
                pca = PCA(n_components=self.n_comps)
                if self.normalize:
                    xy = np.concatenate( (xs_nrm, yt_nrm), axis=0 )
                else:
                    xy = np.concatenate( (xs, yt), axis=0 )
                xy = pca.fit_transform(xy)
                xs_red, yt_red = xy[:n], xy[n:]
            if self.method == 'gromov' and Mx is None:
                self._log("Computing source PC view, %i PCs..."\
                          % self.n_comps)
                pca = PCA(n_components=self.n_comps)
                xs_red = pca.fit_transform(
                    xs_nrm if self.normalize else xs
                )
            if self.method == 'gromov' and My is None:
                self._log("Computing reference PC view, %i PCs..."\
                          % self.n_comps)
                pca = PCA(n_components=self.n_comps)
                yt_red = pca.fit_transform(
                    yt_nrm if self.normalize else yt
                )

        wx, wy = None, None
        if is_label_weighted:
            self._log("Using labels to balance class weights.")
            wx, wy = weight_per_label(xs_labels, yt_labels)

        # Building full TDatas using computed values
        self.tdata_x = TData(xs,
                             x_nrm=xs_nrm,
                             x_red=xs_red,
                             weighted=self.weighted,
                             weights=wx,
                             metric=self.metric,
                             scale=self.scale,
                             alpha_qp=self.alpha_qp,
                             verbose=self.verbose)

        self.tdata_y = TData(yt,
                             x_nrm=yt_nrm,
                             x_red=yt_red,
                             weighted=self.weighted,
                             weights=wy,
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
            if is_label_weighted:
                Mxy = penalize_per_label(Mxy, xs_labels, yt_labels)
        if self.method == 'gromov' and Mx is None:
            self._log("Using metric %s as a cost for Mx. Normalization: %r" %
                     (self.metric, self.normalize))
            Mx = self.tdata_x.distance()
        if self.method == 'gromov' and My is None:
            self._log("Using metric %s as a cost for My. Normalization: %r" %
                     (self.metric, self.normalize))
            My = self.tdata_y.distance()

        # Projecting source to ref
        self._log("Computing transport plan (%s)..." % self.method)
        Pxy = compute_transport(
            self.tdata_x.weights(),
            self.tdata_y.weights(),
            method=self.method,
            Mxy=Mxy,
            Mx=Mx,
            My=My,
            max_iter=self.max_iter,
            entropy=self.entropy,
            hreg=self.hreg,
            unbalanced=self.unbalanced,
            mreg=self.mreg)

        # Anticipating multi-batches
        self.transports.append(Transport(self.tdata_x, self.tdata_y, Pxy))

        self.fitted = True
        self._log("Transmorph fitted.")


    def transform(self,
                  xs: np.ndarray = None,
                  jitter: bool = True,
                  jitter_std: float = .01) -> np.ndarray:
        """
        Applies optimal transport integration. Transmorph must be fitted beforehand.

        xs: np.ndarray, default = None
            Source dataset to transform.

        jitter: bool, default = True
            Adds a little bit of random jittering to the final results. Helps
            downstream methods such as UMAP to converge in some cases.

        jitter_std: float, default = 0.01
            Jittering standard deviation.

        Returns:
        --------
        (n,d1) np.ndarray, of xs integrated onto yt.
        """
        assert self.fitted, "Transmorph must be fitted first."
        self._log("Projecting dataset...")
        xt = transform(self.transports[0],
                       jitter=jitter,
                       jitter_std=jitter_std)
        self._log("Terminated.")
        return xt


    def fit_transform(self,
                      xs: np.ndarray,
                      yt: np.ndarray,
                      xs_labels: np.ndarray = None,
                      yt_labels: np.ndarray = None,
                      Mx: np.ndarray = None,
                      My: np.ndarray = None,
                      Mxy: np.ndarray = None,
                      jitter: bool = True,
                      jitter_std: float = .01) -> np.ndarray:
        """
        Shortcut, fit() -> transform()
        """
        self.fit(xs,
                 yt,
                 xs_labels=xs_labels,
                 yt_labels=yt_labels,
                 Mx=Mx,
                 My=My,
                 Mxy=Mxy)
        return self.transform(xs,
                              jitter=jitter,
                              jitter_std=jitter_std)


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
        assert len(self.transports) == 1, "Subsampling unsupported yet."
        return y_labels[np.argmax(self.transports[0].P.toarray(), axis=1)]
