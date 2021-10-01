#!/usr/bin/env python3

import numpy as np

from collections import namedtuple
from sklearn.utils import check_array
from warnings import warn

from .constants import *
from .integration import compute_transport
from .integration import (
    project,
    transform_reference_space,
    transform_latent_space
)
from .tdata import TData


# TODO: better cacheing


weighting_strategies = [
    "custom",
    "labels",
    "uniform",
    "woti"
]
# Read-only container for transport results
# TData, TData, array
Transport = namedtuple("Transport", ["src", "tar", "P"])


class Transmorph:
    """
    Optimal transport-related tools for dataset integration. This is the main
    class.

    This class implements a set of methods related to optimal transport-based
    unsupervised and semi-supervised learning, with the option of using an
    unsupervised heuristic to estimate data points weights prior to optimal
    transport in order to correct for dataset classes unbalance.

    By default, the optimal transport plan will be computed using
    optimal transport metric, but it can be changed to rather use Gromov-
    Wasserstein distance.

    Transmorph is built on top of the package POT::

        https://pythonot.github.io/

    Parameters
    ----------

    method: str in ('ot', 'gromov'), default = 'ot'
        Transportation framework for data integration.

        OT stands for optimal transport, and requires to define a metric
        between points in different distributions. It is in general
        more reliable in the case of datasets presenting symmetries.

        Gromov stands for Gromov-Wasserstein (GW), and only requires the metric
        in both spaces. It is in this sense more general, but is
        invariant to dataset isometries which can lead to poor
        integration due to overfitting.

    latent_space: bool, default = False
        Integrate the data in a UMAP-like latent space. This allows for
        integration without a reference, at a cost of interpretabiltity
        and distortion due to the low dimensional representation.
    
    metric: str, default = 'sqeuclidan'
        Metric to use when computing cost matrix, must be a string or a
        callable, must be scipy-compatible. For a comprehensive list of
        metrics, see::

            https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

    metric_kwargs: dict, default = {}
        Optional parameters for argument $metric, passed to cdist. 

    geodesic: bool, default=False
        Only available for method = 'gromov'. Turns on geodesic distance on
        a graph endowed with specified metric along its edges instead of vector-based
        metric.

        WARNING: This option is computationally intensive. If performance is
        an issue, we recommend turning on the n_hops parameter.

    normalize: bool, default=True
        Column-normalize matrices before computing cost matrix. Useful when features
        present different scales, and an Euclidean-based metric is used.

    n_comps: int, default=-1
        Number of dimensions to use in the PCA, which is used to compute the cost
        matrix/density. -1 uses the whole feature space.

    entropy: bool, default=False
        Enables the entropy regularization for OT/GW problem, solving it
        approximately but more efficiently using Sinkhorn's algorithm (Cuturi 2013).

    hreg: float, default=1e-3
        Entropic regularization parameter. The lower the more accurate the result
        will be, at a cost of extra computation time. Making hreg too low usually
        results in convergence issues.

    unbalanced: bool, default=False
        Use the unbalanced optimal transport formulation with an entropy regularization.
        Incompatible with Gromov-Wasserstein. Can help when dealing with datasets
        presenting class imbalance.

    mreg: float, default=1e-3
        Regularization parameter for unbalanced formulation. Needs tuning, otherwise
        leading to convergence issues.

    weighting_strategy: str, default='uniform'
        Strategy to use in order to reweight the samples, before optimal transport.
        Possible values are:
    
            - 'uniform', all points have the same mass
            - 'woti', automatic weights selection based on local density
            - 'labels', automatic weights selection based on labels
            - 'custom', provide custom weights to the fit() function
    
    label_dependency: float, default=0
        Adds a label dependency in the cost matrix, in the case of supervised optimal
        transport formulation. Can be tuned between 0 (cost matrix does not include
        label penalty) and 1 (label penalty dominates data geometry). If set to a nonzero
        value, you must pass samples labels to the fit() function.

    n_hops: int, default=0
        Increase n_hops to compute OT on a subsample of data, which increases the
        computational efficiency but decreases result accuracy. If n_hops = 0, then
        all points are used during optimal transport (exact solution).

    max_iter: int, default = 1e2
        Maximum number of iterations for OT/GW.

    low_memory: bool, default = False
        Trades computation time for lower memory usage. We plan on making
        it influence more pipeline parts in the future.

    verbose: int, default = 1
        Defines the logging level.
        0: Disabled
        1: Informations
        2: Debug

    Example
    -------
    >>> from transmorph.datasets import load_spirals
    >>> import transmorph as tr
    >>> x, y = load_spirals()
    >>> x.shape, y.shape
    ((433, 3), (633, 3))
    >>> my_transmorph = tr.Transmorph(method='ot')
    >>> x_integrated = my_transmorph.fit_transform(x, y)
    """

    def __init__(
            self,
            method: str = "ot",
            same_space_input: bool = True,
            latent_space: bool = False,
            latent_dim: int = 2,
            umap_kwargs: dict = {},
            metric: str = "sqeuclidean",
            metric_kwargs: dict = {},
            geodesic: bool = False,
            normalize: bool = False,
            n_comps: int = -1,
            n_neighbors: int = 15,
            entropy: bool = False,
            hreg: float = 1e-3,
            unbalanced: bool = False,
            mreg: float = 1e-3,
            weighting_strategy: str = "uniform",
            label_dependency: float = 0,
            n_hops: int = 0,
            max_iter: int = None,
            low_memory: bool = False,
            random_seed: int = 42,
            verbose: int = 1
    ):
        self.metric = metric
        self.same_space_input = same_space_input
        self.latent_space = latent_space
        self.latent_dim = latent_dim
        self.umap_kwargs = umap_kwargs
        self.metric_kwargs = metric_kwargs
        self.geodesic = geodesic
        self.normalize = normalize
        self.n_comps = n_comps
        self.n_neighbors = n_neighbors
        self.entropy = entropy
        self.hreg = hreg
        self.unbalanced = unbalanced
        self.mreg = mreg
        self.method = method
        self.weighting_strategy = weighting_strategy
        self.label_dependency = label_dependency
        self.n_hops = n_hops
        self.max_iter = max_iter
        self.low_memory = low_memory
        self.random_seed = random_seed
        self.verbose = verbose

        self.transport = None
        self.fitted = False

        self.validate_parameters()

        self._log("Transmorph successfully initialized.", level=1)
        self._log(str(self), level=2)


    def validate_parameters(self):

        ## Raw checks

        # Method
        assert self.method in ("ot", "gromov"),\
            f"Unrecognized method: {self.method}. Available methods "\
            "are 'ot', 'gromov'."

        # Conversion to integer constants
        if self.method == "gromov":
            self.method = TR_METHOD_GROMOV
        elif self.method == "ot":
            self.method = TR_METHOD_OT
        else:
            raise NotImplementedError

        # Metric
        assert callable(self.metric) or isinstance(self.metric, str),\
            "Unrecognized metric type. Must be a callable or a string."
        
        # Geodesic only makes sense if method is gromov
        if self.geodesic:
            assert self.method == TR_METHOD_GROMOV,\
                "geodesic=True only available if method 'gromov'."

        # Valid number of components
        assert isinstance(self.n_comps, int) and self.n_comps >= -1,\
            f"Unrecognized number of components for PCA: {self.n_comps}."

        # Valid entropy regularizer
        if isinstance(self.hreg, int):
            self.hreg = float(self.hreg)
        assert isinstance(self.hreg, float) and self.hreg > 0,\
            f"Entropy regularizer hreg must be positive, found {self.hreg}."

        # Valid marginal penalty
        if isinstance(self.mreg, int):
            self.mreg = float(self.mreg)
        assert isinstance(self.mreg, float) and self.mreg > 0,\
            f"Marginal penalty mreg must be positive, found {self.mreg}."
        
        # Valid weighting strategy
        assert self.weighting_strategy in weighting_strategies,\
            f"Unrecognized weighting strategy: {self.weighting_strategy}. "\
            f"Recognized strategies: {','.join(weighting_strategies)}"
        if self.weighting_strategy == "uniform":
            self.weighting_strategy = TR_WS_UNIFORM
        elif self.weighting_strategy == "woti":
            self.weighting_strategy = TR_WS_AUTO
        elif self.weighting_strategy == "labels":
            self.weighting_strategy = TR_WS_LABELS
        elif self.weighting_strategy == "custom":
            self.weighting_strategy = TR_WS_CUSTOM
        else:
            raise NotImplementedError

        # Valid label dependency
        if isinstance(self.label_dependency, int):
            self.label_dependency = float(self.label_dependency)
        assert isinstance(self.label_dependency, float)\
                and 0 <= self.label_dependency <= 1,\
            f"Invalid label dependency coefficient: {self.label_dependency}, "\
            "expected to be between 0 and 1."

        # Valid number of hops
        assert isinstance(self.n_hops, int) and self.n_hops >= 0,\
            f"Invalid number of hops: {self.n_hops}."
       
        # Valid number of iterations
        if self.max_iter is None:
            if self.entropy:  # Log-stabilized is expansive
                self.max_iter = 5e2
            else:
                self.max_iter = 1e6
        if isinstance(self.max_iter, float):
            self.max_iter = int(self.max_iter)
        assert isinstance(self.max_iter, int) and self.max_iter > 0,\
            f"Invalid maximum number of iterations: {self.max_iter}."

        # Valid level of verbose
        if isinstance(self.verbose, bool):
            self.verbose = 2*self.verbose
        assert isinstance(self.verbose, int) and 0 <= self.verbose < 3,\
            f"Invalid level of verbose: {self.verbose}"

        # Combination checks
        assert not (
            self.method == TR_METHOD_GROMOV
            and self.label_dependency > 0), \
            "Labels dependency is incompatible with Gromov-Wasserstein."

        assert not (
            self.method == TR_METHOD_GROMOV
            and self.unbalanced == True), \
            "Unbalanced is incompatible with Gromov-Wasserstein."


    def __str__(self) -> str: # TODO: rework this, making it more concise
        return \
        "<Transmorph> object.\n"\
        f"-- latent_space: {self.latent_space}\n"\
        f"-- method: {self.method}\n"\
        f"-- metric: {self.metric}\n"\
        f"-- geodesic: {self.geodesic}\n"\
        f"-- normalize: {self.normalize}\n"\
        f"-- n_comps: {self.n_comps}\n"\
        f"-- entropy: {self.entropy}\n"\
        f"-- hreg: {self.hreg}\n"\
        f"-- unbalanced: {self.unbalanced}\n"\
        f"-- mreg: {self.mreg}\n"\
        f"-- weighting_strategy: {self.weighting_strategy}\n"\
        f"-- label_dependency: {self.label_dependency}\n"\
        f" -- n_hops: {self.n_hops}\n"\
        f"-- max_iter: {self.max_iter}\n"\
        f"-- verbose: {self.verbose}\n"\
        f"-- fitted: {self.fitted}"

    
    def _log(
        self,
        s: str,
        end: str = "\n",
        header: bool = True,
        level=2
    ) -> None:
        # Internal logginf method
        # Only prints for now, can later be pipelined into other streams
        if level > self.verbose:
            return
        if header:
            s = f"# Transmorph > {s}"
        print(s, end=end)


    def fit(
        self,xs: np.ndarray,
        yt: np.ndarray,
        xs_labels: np.ndarray = None,
        yt_labels: np.ndarray = None,
        xs_weights: np.ndarray = None,
        yt_weights: np.ndarray = None,
        Mx: np.ndarray = None,
        My: np.ndarray = None,
        Mxy: np.ndarray = None
    ) -> None:
        """
        Computes the optimal transport plan between two empirical distributions,
        with parameters specified during initialization. Caches the result
        in the Transmorph object.

        Parameters:
        -----------
        xs: (n,d0) np.ndarray
            Query data points cloud.

        yt: (m,d1) np.ndarray
            Target data points cloud.

        xs_labels: (n,) np.ndarray, default=None
            Optional parameter, for supervised cases.
            Labels of xs points, for the supervised integration.

        yt_labels: (m,) np.ndarray, default=None
            Optional parameter, for supervised cases.
            Label of yt points, for the supervised integration.

        xs_weights: (n,) np.ndarray, default=None
            Weights of query dataset points. If None, weights are
            inferred with respect to the selected weighting_strategy.

        yt_weights: (m,) np.ndarray
            Weights of reference dataset points. If None, weights are
            inferred with respect to the selected weighting_strategy.

        Mx: (n,n) np.ndarray
            Pairwise metric matrix for xs. Only relevant for GW. If
            None, selected metric is used.

        My: (m,m) np.ndarray
            Pairwise metric matrix for yt. Only relevant for GW. If
            None, selected metric is used.

        Mxy: (n,m) np.ndarray
            Pairwise metric matrix between xs and yt. Only relevant for OT.
            If None, selected metric distance is used.
        """
        ### Verifying parameters ###

        # Data matrices
        n, m = len(xs), len(yt)
        assert n > 0, "Empty query matrix."
        assert m > 0, "Empty reference matrix."

        xs = check_array(xs, dtype=np.float64, order="C")
        yt = check_array(yt, dtype=np.float64, order="C")

        _labels_necessary = (
            self.weighting_strategy == TR_WS_LABELS
            or self.label_dependency > 0
        )
        # What to do with labels
        if _labels_necessary:
            assert xs_labels is not None,\
                "Label-based weighting strategy, but no query labels."
            assert yt_labels is not None,\
                "Label-based weighting strategy, but no reference labels."

        # Verifying labels size
        if xs_labels is not None:
            assert xs_labels.shape[0] == n,\
                "Inconsistent size for labels in query dataset, "\
                f"{xs_labels.shape[0]} != {n}"
        if yt_labels is not None:
            assert yt_labels.shape[0] == m,\
                "Inconsistent size for labels in reference dataset, "\
                f"{yt_labels.shape[0]} != {m}"

        # Verifying user custom weights
        if xs_weights is not None:
            assert xs_weights.shape[0] == n,\
                "Inconsistent dimension for weights in query dataset, "\
                f"{xs_weights.shape[0]} != {n}"
        if yt_weights is not None:
            assert yt_weights.shape[0] == m,\
                "Inconsistent dimension for weights in reference dataset, "\
                f"{yt_weights.shape[0]} != {m}"

        # Verifying user custom cost matrices
        if self.method == TR_METHOD_OT:
            assert Mx is None and My is None,\
                "Method is optimal transport, but dataset-specific "\
                "metrics have been set. See parameter Mxy instead."
        if self.method == TR_METHOD_GROMOV:
            assert Mxy is None,\
                "Method is Gromov-Wasserstein, but cross-dataset "\
                "metric have been set. See parameters Mx and My instead."
        if self.method == TR_METHOD_OT and Mxy is not None:
            Mxy = check_array(Mxy, dtype=np.float32, order="C")
            assert Mxy.shape == (n, m), \
                "Inconsistent dimension between user-provided cost "\
                f"matrix and datasets size. Expected: ({n},{m}), "\
                f"found: {Mxy.shape}."
        if self.method == TR_METHOD_GROMOV and Mx is not None:
            Mx = check_array(Mx, dtype=np.float32, order="C")
            assert Mx.shape == (n, n), \
                "Inconsistent dimension between user-provided metric "\
                f"matrix and query dataset size. Expected: ({n},{n}), "\
                f"found: {Mx.shape}."
        if self.method == TR_METHOD_GROMOV and My is not None:
            My = check_array(My, dtype=np.float32, order="C")
            assert My.shape == (m, m), \
                "Inconsistent dimension between user-provided metric "\
                f"matrix and reference dataset size. Expected: ({m},{m}), "\
                f"found: {My.shape}."

        self._log("Parameters checked.", level=2)
        ### Starting the procedure ###

        self.fitted = False

        # Building full TDatas using computed values
        self._log("Creating TDatas...", level=2)
        self.tdata_x = TData(
            xs,
            metric=self.metric,
            metric_kwargs=self.metric_kwargs,
            geodesic=self.geodesic,
            normalize=self.normalize,
            n_comps=self.n_comps,
            custom_pca=None,
            n_neighbors=self.n_neighbors,
            n_hops=self.n_hops,
            weighting_strategy=self.weighting_strategy, #FIXME: type jumping
            weights=xs_weights,
            labels=xs_labels,
            low_memory=self.low_memory,
            verbose=self.verbose
        )

        custom_pca = None
        if self.n_comps > 0 and self.method == TR_METHOD_OT:
            custom_pca = self.tdata_x.get_extra("pca")

        self.tdata_y = TData(
            yt,
            metric=self.metric,
            metric_kwargs=self.metric_kwargs,
            geodesic=self.geodesic,
            normalize=self.normalize,
            n_comps=self.n_comps,
            custom_pca=custom_pca,
            n_neighbors=self.n_neighbors,
            n_hops=self.n_hops,
            weighting_strategy=self.weighting_strategy,
            weights=yt_weights,
            labels=yt_labels,
            low_memory=self.low_memory,
            verbose=self.verbose
        )

        subsample = self.n_hops > 0

        # Computing cost matrices if necessary.
        if self.method == TR_METHOD_OT and Mxy is None:
            Mxy = self.tdata_x.distance(
                other=self.tdata_y,
                subsampling=subsample,
            )
            if self.label_dependency:
                assert xs_labels is not None, "No labels found for x."
                assert yt_labels is not None, "No labels found for y."
                label_discrepancy = (
                    xs_labels[:,None] != yt_labels
                )
                Mxy = (
                    Mxy
                    + self.label_dependency * Mxy.max() * label_discrepancy
                )
        if self.method == TR_METHOD_GROMOV and Mx is None:
            Mx = self.tdata_x.distance(
                subsampling=subsample,
            )
        if self.method == TR_METHOD_GROMOV and My is None:
            My = self.tdata_y.distance(
                subsampling=subsample,
            )

        if self.weighting_strategy == TR_WS_LABELS:
            self.tdata_x.compute_weights(other_if_labels=self.tdata_y)
        else:
            self.tdata_x.compute_weights()
            self.tdata_y.compute_weights()

        wx = self.tdata_x.get_attribute("weights", subsample=subsample)
        wy = self.tdata_y.get_attribute("weights", subsample=subsample)

        # Ensuring weights is in the simplex
        wx /= wx.sum()
        wy /= wy.sum()

        # Projecting query to ref
        self._log("Computing transport plan...", level=1)
        Pxy = compute_transport(
            wx,
            wy,
            method=self.method,
            Mxy=Mxy,
            Mx=Mx,
            My=My,
            max_iter=self.max_iter,
            entropy=self.entropy,
            hreg=self.hreg,
            unbalanced=self.unbalanced,
            mreg=self.mreg
        )

        self.transport = Transport(self.tdata_x, self.tdata_y, Pxy)

        self.fitted = True
        self._log("Transmorph fitted.", level=1)


    def transform(
        self,
        jitter: bool = True,
        jitter_std: float = .01
    ) -> np.ndarray:
        """
        Applies optimal transport integration. Transmorph must be fitted beforehand.

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

        subsample = self.n_hops > 0
        P = self.transport.P
        tdata_x, tdata_y = self.tdata_x, self.tdata_y
        get_attr_X = tdata_x.get_attribute
        get_attr_Y = tdata_y.get_attribute
        X, Y = tdata_x.X, tdata_y.X
        x_weights, y_weights = (
            get_attr_X("weights", subsample=subsample),
            get_attr_Y("weights", subsample=subsample)
        )
        x_anchors, y_anchors = (
            get_attr_X("anchors", subsample=False),
            get_attr_Y("anchors", subsample=False)
        )
        x_mapping, y_mapping = (
            get_attr_X("mapping", subsample=False),
            get_attr_Y("mapping", subsample=False)
        )
        
        subsample = self.n_hops > 0
        if self.latent_space:

            # Within-dataset smoothed representations
            tdata_x.smooth_by_pooling()
            tdata_y.smooth_by_pooling()
            X1 = tdata_x.get_layer("pooled", subsample=subsample)
            X2 = tdata_y.get_layer("pooled", subsample=subsample)

            self._log("Computing reciprocal projections...")
            # Project X1 in S2
            X12 = transform_reference_space(
                P,
                X1,
                x_weights,
                x_anchors[x_anchors],
                x_mapping[x_anchors],
                X2,
                y_weights,
                y_anchors[y_anchors],
                y_mapping[y_anchors],
                continuous_displacement=False
            )
            
            # Project X2 in S1
            X21 = transform_reference_space(
                P.T,
                X2,
                y_weights,
                y_anchors[y_anchors],
                y_mapping[y_anchors],
                X1,
                x_weights,
                x_anchors[x_anchors],
                x_mapping[x_anchors],
                continuous_displacement=False
            )

            self._log("Building joint graphs...")
            # Build joint graphs
            tdata_x12 = TData(
                np.concatenate( (X12, X2), axis=0 ),
                metric=self.metric,
                metric_kwargs=self.metric_kwargs,
                geodesic=False,
                normalize=self.normalize,
                n_comps=-1,
                custom_pca=None,
                n_neighbors=self.n_neighbors,
                n_hops=0,
                weighting_strategy="uniform",
                weights=None,
                labels=None,
                low_memory=self.low_memory,
                verbose=self.verbose
            )
            tdata_x12.neighbors()

            tdata_x21 = TData(
                np.concatenate( (X1, X21), axis=0 ),
                metric=self.metric,
                metric_kwargs=self.metric_kwargs,
                geodesic=False,
                normalize=self.normalize,
                n_comps=-1,
                custom_pca=None,
                n_neighbors=self.n_neighbors,
                n_hops=0,
                weighting_strategy="uniform",
                weights=None,
                labels=None,
                low_memory=self.low_memory,
                verbose=self.verbose
            )
            tdata_x21.neighbors()

            self._log("Embedding in latent space...")
            # Compute embedding
            xt = transform_latent_space(
                tdata_x.get_extra("strength_graph"),
                tdata_y.get_extra("strength_graph"),
                tdata_x12.get_extra("strength_graph"),
                tdata_x21.get_extra("strength_graph"),
                x_anchors,
                y_anchors,
                latent_dim=self.latent_dim,
                umap_kwargs=self.umap_kwargs,
                n_neighbors=self.n_neighbors,
                metric=self.metric,
                metric_keywords=self.metric_kwargs,
                random_state=self.random_seed
            )

        else:

            if (self.same_space_input is False
                and self.n_hops > 0):
                warn("Integrating datasets from different spaces with "\
                     "subsampling enabled without latent space embedding "\
                     "is discouraged. Only anchors will be integrated. "\
                     "Try turning latent_space=True.")
                
            xt = transform_reference_space(
                P,
                X,
                x_weights,
                x_anchors,
                x_mapping,
                Y,
                y_weights,
                y_anchors,
                y_mapping,
                continuous_displacement=self.same_space_input
            )

        assert not np.isnan(np.sum(xt)),\
            "Integrated matrix contains NaN values. Please ensure the input "\
            "is correct, and try increasing 'hreg' and 'mreg' parametrs."


        if jitter:
            stdev = jitter_std * (
                np.max(xt, axis=0) - np.min(xt, axis=0)
            )
            xt += np.random.randn(*xt.shape) * stdev

        self._log("Terminated.")
        return xt


    def fit_transform(
        self,
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
        jitter_std: float = .01
    ) -> np.ndarray:
        """
        Shortcut, fit() -> transform()
        """
        self.fit(
            xs,
            yt,
            xs_labels=xs_labels,
            yt_labels=yt_labels,
            xs_weights=xs_weights,
            yt_weights=yt_weights,
            Mx=Mx,
            My=My,
            Mxy=Mxy
        )
        return self.transform(
            jitter=jitter,
            jitter_std=jitter_std
        )


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
        (n,) np.ndarray, predicted labels for query dataset points.
        """
        assert self.fitted, "Transmorph must be fitted first."

        Pxy = self.transport.P.toarray()
        assert len(y_labels) == Pxy.shape[1],\
            "Inconsistent size between fitted $ys and $y_labels. "\
            f"Expected ({Pxy.shape[1]}), found ({len(y_labels)})."

        return y_labels[np.argmax(Pxy, axis=1)]


    def wasserstein_distance_transmorph(self):
        """
        Returns the total cost of transport matrix from a fitted
        Transmorph. 
        """
        assert self.n_hops == -1,\
            "Error: Wasserstein distance cannot be computed "\
            "on a subsampled transmorph. Use wasserstein_distance "\
            "instead."
        assert self.fitted,\
            "Error: Transmorph is not fitted."

        return np.sum(self.transport.P)
