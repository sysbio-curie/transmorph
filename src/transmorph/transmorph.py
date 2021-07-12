#!/usr/bin/env python3

import numpy as np

from collections import namedtuple
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.utils import check_array

from .constants import *
from .integration import compute_transport
from .integration import transform
from .tdata import TData
from .utils import col_normalize

# TODO: better cacheing

_tmp_aliases_methods = [
    ['woti', 'auto', 'automatic', 'qp'],
    ['labeled', 'labels'],
    ['uniform', 'none', 'nil', 'uni']
]
_aliases_methods = {
    k: i
    for (i, keys) in enumerate(_tmp_aliases_methods)
    for k in keys
}

# Read-only container for transport results
# TData, TData, array
Transport = namedtuple("Transport", ["src", "tar", "P"])

def penalize_per_label(C, xs_labels, yt_labels, lam=1):
    """
    Defines a label-dependent cost matrix for supervised problems
    f(C, xl, yl, lam)_ij = C_ij + lam * C.max * (xli == ylj)

    Parameters:
    -----------

    C: (n,m) np.ndarray
        Raw cost matrix (e.g. Euclidean)

    xs_labels: (n,1) np.ndarray
        Source dataset labels

    yt_labels: (m,1) np.ndarray
        Reference dataset label

    lam: float, default = 1
        for lam \in [0,1], lam being the label dependency factor
        lam=0 -> C remains unchanged
        lam=1 -> C is label-consistent

    Returns:
    --------

    (n,m) np.ndarray
        Label-corrected cost matrix.

    """
    return C + lam*C.max()*(xs_labels[:,None] != yt_labels)


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

    metric: str, default = 'sqeuclidan'
        Default metric to use. (see scipy.spatial.distance.cdist)

    geodesic: bool, default = False
        Only available for method = 'gromov'. Use geodesic
        distance instead on a graph instead of vector-based metric.

    normalize: bool = 1
        Column-normalize matrices before computing cost matrix.

    n_comps: int, default = -1
        Number of dimensions to use in the PCA, which is used to compute the cost
        matrix/density. -1 uses the raw dimensionality instead.

    entropy: bool, default = False
        Enables the entropy regularization for OT/GW problem, solving it
        approximately but more efficiently using Sinkhorn's algorithm
        (Cuturi 2013)

    hreg: float, default = 1e-3
        Entropic regularization parameter. The lower the more accurate the result
        will be, at a cost of extra computation time.

    unbalanced: bool, default = False
        Use the unbalanced optimal transport formulation + entropy regularization.
        Incompatible with Gromov-Wasserstein

    mreg: float, default = 1e-3
        Regularization parameter for unbalanced formulation.

    weighting_strategy: str, default = uniform
        Strategy to use in order to reweight the samples, before optimal transport.
        Possible values are:
        - 'uniform', all points have the same mass
        - 'woti', automatic weights selection based on local density
        - 'labels', automatic weights selection based on labels
    
    label_dependency: float, default = 0
        In the case of known labels, extra penalty in the cost matrix if
        labels differ.

    n_hops: int, default = 0
        Number of hops to consider for the vertex cover approximation.
        If n_hops = 0, then all points are used during optimal transport (exact
        solution).

    max_iter: int, default = 1e6
        Maximum number of iterations for OT/GW.

    verbose: int, default = 1
        Defines the logging level.
        0: Disabled
        1: Informations
        2: Debug
    """

    def __init__(self,
                 method: str = 'ot',
                 metric: str = 'sqeuclidean',
                 geodesic: bool = False,
                 normalize: bool = False,
                 n_comps: int = -1,
                 entropy: bool = False,
                 hreg: float = 1e-3,
                 unbalanced: bool = False,
                 mreg: float = 1e-3,
                 weighting_strategy: str = 'uniform',
                 label_dependency: float = 0,
                 n_hops: int = 0,
                 max_iter: int = 1e6,
                 verbose: bool = False):

        self.method = method
        self.metric = metric
        self.geodesic = geodesic
        self.normalize = normalize
        self.n_comps = n_comps
        self.entropy = entropy
        self.hreg = hreg
        self.unbalanced = unbalanced
        self.mreg = mreg
        self.weighting_strategy = weighting_strategy
        self.label_dependency = label_dependency
        self.n_hops = n_hops
        self.max_iter = int(max_iter)
        self.verbose = verbose

        self.transport = None
        self.fitted = False

        self.validate_parameters()

        self._log("Successfully initialized.")
        self._log(str(self), level=2)


    def validate_parameters(self):

        # Raw checks
        assert self.method in ('ot', 'gromov'), \
            "Unrecognized method: %s. Available methods \
            are 'ot', 'gromov'" % self.method

        if self.method == 'gromov':
            self.method = TR_METHOD_GROMOV
        elif self.method == 'ot':
            self.method = TR_METHOD_OT
        else:
            raise NotImplementedError

        # geodesic => gromov
        if self.geodesic:
            assert self.method == TR_METHOD_GROMOV, \
                "geodesic = True only available for method 'gromov'."

        assert self.weighting_strategy in _aliases_methods, \
            "Unrecognized weighting strategy: %s. Available \
            strategies: %s" \
            % (self.weighting_strategy, ','.join(_aliases_methods.keys()))

        self.weighting_strategy = _aliases_methods[self.weighting_strategy]

        assert 0 <= self.label_dependency <= 1, \
            "Invalid label dependency coefficient: %f, expected \
            to be between 0 and 1."

        assert self.max_iter > 0, \
            "Negative number of iterations."

        # Combination checks
        assert not (
            self.method == TR_METHOD_GROMOV
            and self.weighting_strategy == TR_WS_LABELS), \
            "Labels weighting is incompatible with Gromov-Wasserstein."

        assert not (
            self.method == TR_METHOD_GROMOV
            and self.unbalanced == True), \
            "Unbalanced is incompatible with Gromov-Wasserstein."


    def __str__(self) -> str:
        # Useful for debugging
        return f'\n\
        <Transmorph> object.\n\
        -- method: {self.method}\n\
        -- metric: {self.metric}\n\
        -- geodesic: {self.geodesic}\n\
        -- normalize: {self.normalize}\n\
        -- n_comps: {self.n_comps}\n\
        -- entropy: {self.entropy}\n\
        -- hreg: {self.hreg}\n\
        -- unbalanced: {self.unbalanced}\n\
        -- mreg: {self.mreg}\n\
        -- weighting_strategy: {self.weighting_strategy}\n\
        -- label_dependency: {self.label_dependency}\n\
        -- n_hops: {self.n_hops}\n\
        -- max_iter: {self.max_iter}\n\
        -- verbose: {self.verbose}\n\
        -- fitted: {self.fitted}\n\
        '

    def _log(self, s: str, end: str = '\n', header: bool = True, level=2) -> None:
        # Internal logginf method
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
            xs_weights: np.ndarray = None,
            yt_weights: np.ndarray = None,
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
            Labels of xs points, for the supervised integration.

        yt_labels: (m,) np.ndarray
            Label of yt points, for the supervised integration.

        xs_weights: (n,) np.ndarray
            Weights of source dataset points.

        yt_weights: (m,) np.ndarray
            Weights of reference dataset points.

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

        ### Verifying parameters ###

        # Data matrices
        n, m = len(xs), len(yt)
        assert n > 0, "Empty source matrix."
        assert m > 0, "Empty reference matrix."

        xs = check_array(xs, dtype=np.float32, order="C")
        yt = check_array(yt, dtype=np.float32, order="C")

        if xs_weights is not None:
            xs_weights = check_array(xs_weights, dtype=np.float32, order="C")

        if yt_weights is not None:
            yt_weights = check_array(yt_weights, dtype=np.float32, order="C")

        # Use sample labels frequency to estimate weights?
        is_label_weighted = (
            xs_labels is not None
            and yt_labels is not None
            and xs_weights is None
            and yt_weights is None
        )

        assert not (
            is_label_weighted
            and self.weighting_strategy == TR_WS_AUTO), \
            "Inconsistency in parameters: Automatic weighting chosen, \
            but labels were passed to fit()."

        assert not (
            not is_label_weighted
            and self.weighting_strategy == TR_WS_LABELS), \
            "Inconsistency in parameters: Labels weighting chosen, \
            but no labels were passed to fit()."

        # Verifying labels
        if is_label_weighted:
            assert xs_labels.shape[0] == n, \
                "Inconsistent dimension for labels in source dataset, \
                %i != %i" % (xs_labels.shape[0], n)
            assert yt_labels.shape[0] == m, \
                "Inconsistent dimension for labels in reference dataset, \
                %i != %i" % (yt_labels.shape[0], m)

        # Verifying label dependency
        if self.label_dependency:
            assert self.method == TR_METHOD_OT, \
                "Label dependency cannot be used with methods other \
                than optimal transport."
            assert is_label_weighted, "Label dependency cannot be \
                applied without labels."

        # Verifying weights
        if xs_weights is not None:
            assert xs_weights.shape[0] == n, \
                "Inconsistent dimension for weights in source dataset, \
                %i != %i" % (xs_weights.shape[0], n)

        if yt_weights is not None:
            assert yt_weights.shape[0] == m, \
                "Inconsistent dimension for weights in reference dataset, \
                %i != %i" % (yt_weights.shape[0], m)

        # Cost matrices
        if self.method == TR_METHOD_OT and Mxy is not None:
            Mxy = check_array(Mxy, dtype=np.float32, order="C")
            assert Mxy.shape == (n, m), \
                "Inconsistent dimension between user-provided cost \
                matrix and datasets size. Expected: (%i,%i), found: (%i,%i)" \
                % (n, m, *Mxy.shape)
        if self.method == TR_METHOD_GROMOV and Mx is not None:
            Mx = check_array(Mx, dtype=np.float32, order="C")
            assert Mx.shape == (n, n), \
                "Inconsistent dimension between user-provided cost \
                matrix and source dataset size. \
                Expected: (%i,%i), found: (%i,%i)" \
                % (n, n, *Mx.shape)
        if self.method == TR_METHOD_GROMOV and My is not None:
            My = check_array(My, dtype=np.float32, order="C")
            assert My.shape == (m, m), \
                "Inconsistent dimension between user-provided cost \
                matrix and reference dataset size. \
                Expected: (%i,%i), found: (%i,%i)" \
                % (m, m, *My.shape)

        ### Parameters verified. ###
        ### Starting the procedure ###

        self.fitted = False

        # Building full TDatas using computed values
        self._log("Creating TDatas...")
        self.tdata_x = TData(xs,
                             weights=xs_weights,
                             labels=xs_labels,
                             normalize=self.normalize,
                             verbose=self.verbose)

        self.tdata_y = TData(yt,
                             weights=yt_weights,
                             labels=yt_labels,
                             normalize=self.normalize,
                             verbose=self.verbose)

        layer = 'raw'
        if self.n_comps != -1:
            layer = 'pca'
            self._log("Computing PCAs, %i dimensions..." % self.n_comps)
            if self.method == TR_METHOD_OT:
                self.tdata_x.pca(n_components=self.n_comps, other=self.tdata_y)
            elif self.method == TR_METHOD_GROMOV:
                self.tdata_x.pca(n_components=self.n_comps)
                self.tdata_y.pca(n_components=self.n_comps)

        # KNN-graph construction for geodesic/vertex cover
        if self.geodesic or self.n_hops:
            self._log("Computing kNN graph...")
            self.tdata_x.neighbors(metric=self.metric,
                                   self_edit=True,
                                   layer=layer)
            self.tdata_y.neighbors(metric=self.metric,
                                   self_edit=True,
                                   layer=layer)

        subsample = self.n_hops > 0
        
        # Vertex cover
        if subsample:
            self._log("Computing %i-hops vertex covers..." % self.n_hops)
            self.tdata_x.select_representers(self.n_hops)
            self._log("Source dataset: %i/%i points kept." %
                (self.tdata_x.anchors.sum(), len(self.tdata_x))
            )
            self.tdata_y.select_representers(self.n_hops)
            self._log("Reference dataset: %i/%i points kept." %
                (self.tdata_y.anchors.sum(), len(self.tdata_y))
            )

        # Weights
        if self.weighting_strategy == TR_WS_AUTO:
            if xs_weights is None:
                self.tdata_x.compute_weights(
                    method=TR_WS_AUTO,
                    subsample=subsample,
                    layer=layer)
            if yt_weights is None:
                self.tdata_y.compute_weights(
                    method=TR_WS_AUTO,
                    subsample=subsample,
                    layer=layer)
        elif self.weighting_strategy == TR_WS_LABELS:
            if (xs_weights is not None
                or yt_weights is not None):
                self._log("Warning: Using labels weighting strategy \
                will override custom weights choice. Consider using \
                'custom' or 'uniform' instead.")
            self.tdata_x.compute_weights(
                method=TR_WS_LABELS,
                layer=layer,
                other=self.tdata_y
            )

        # Computing cost matrices if necessary.
        if self.method == TR_METHOD_OT and Mxy is None:
            self._log("Using metric %s as a cost for Mxy. Normalization: %r" %
                     (self.metric, self.normalize))
            assert xs.shape[1] == yt.shape[1], (
                "Dimension mismatch (%i != %i)" % (xs.shape[1], yt.shape[1]))
            Mxy = self.tdata_x.distance(self.tdata_y,
                                        metric=self.metric,
                                        geodesic=False,
                                        subsample=subsample,
                                        return_full_size=False,
                                        layer=layer)
            if self.label_dependency:
                penalize_per_label(Mxy,
                                   self.tdata_x.labels(),
                                   self.tdata_y.labels(),
                                   self.label_dependency)
        if self.method == TR_METHOD_GROMOV and Mx is None:
            self._log("Using metric %s as a cost for Mx. Normalization: %r" %
                      (self.metric, self.normalize))
            Mx = self.tdata_x.distance(metric=self.metric,
                                       subsample=subsample,
                                       geodesic=self.geodesic,
                                       return_full_size=False,
                                       layer=layer)
        if self.method == TR_METHOD_GROMOV and My is None:
            self._log("Using metric %s as a cost for My. Normalization: %r" %
                     (self.metric, self.normalize))
            My = self.tdata_y.distance(metric=self.metric,
                                       subsample=subsample,
                                       geodesic=self.geodesic,
                                       return_full_size=False,
                                       layer=layer)

        # Projecting source to ref
        self._log("Computing transport plan...")
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

        self.transport = Transport(self.tdata_x, self.tdata_y, Pxy)

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
        assert jitter_std > 0, "Negative standard deviation for jittering."
        self._log("Projecting dataset...")
        xt = transform(self.transport,
                       jitter=jitter,
                       jitter_std=jitter_std)
        self._log("Terminated.")
        return xt


    def fit_transform(self,
                      xs: np.ndarray,
                      yt: np.ndarray,
                      xs_labels: np.ndarray = None,
                      yt_labels: np.ndarray = None,
                      xs_weights: np.ndarray = None,
                      yt_weights: np.ndarray = None,
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

        Pxy = self.transport.P.toarray()
        assert len(y_labels) == Pxy.shape[1], \
            "Inconsistent size between fitted $ys and $y_labels. \
            Expected (%i,), found (%i,)." % (Pxy.shape[1], len(y_labels))

        return y_labels[np.argmax(Pxy, axis=1)]
