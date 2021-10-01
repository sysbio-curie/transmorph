#!/usr/bin/env python3

import numpy as np

from sklearn.decomposition import PCA
from sklearn.utils import check_array
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import dijkstra

from .constants import *
from .density import sigma_search
from .density import normal_kernel_weights
from .utils import (
    col_normalize,
    compute_umap_graph,
    fill_dmatrix,
    symmetrize_distance_matrix,
    symmetrize_strength_matrix,
    vertex_cover,
    weight_per_label,
    within_modality_stabilize
)

# TODO: sliceable
# TODO: sparse support
# TODO: low memory support
#     -> if normalize, only retain normalizing weights

class TData:
    """
    Custom data object used by the main module. Features
    weight computation, subsampling, custom distance, labels handling...

    Parameters
    ----------

    X: (n,d) np.ndarray to embed.

    metric: str or callable, default = "euclidean"
        Scipy-compatible metric to use, will be passed as an argument
        to cdist. Tuning this parameter can be key for method's success.

    metric_kwargs: dct, default = {}
        Additional arguments to pass with metric, only appliable for
        some metrics. See cdist for more info:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

    geodesic: bool, default = False
        Use geodesic distance for within-dataset distance computation.
        Underlying manifold is in this case approximated by the kNN-graph.

    normalize: bool, default = False
        Compute a column-reduced representation of the dataset. It will
        be used by default instead of raw representation in most methods.
        This helps in high dimensional cases with very different features.

    n_comps: int, default = -1
        Number of components in the PC view, set it to -1 to disable PCA
        computation. PC view will be used by default instead of normalized
        representation if available in most methods. This greatly helps
        mitigating curse of dimensionality and performance issues for large
        scale applications.

    n_neighbors: int, default = 15
        Number of neighbors to use in the nearest-neighbors graph, useful
        values typically range from 10 to 50. Notably used for integration
        in a latent space.

    n_hops: int, default = 0
        Number of hops to consider for building the vertex cover, used
        as a way to subsample the dataset. n_hops = 0 disables vertex
        cover. Increasing n_hops greatly improves performance, at an
        accuracy/quality cost; we recommend 1 for large scale applications,
        possibly increasing it to 2 if necessary.

    weighting_strategy: int, default = TR_WS_UNIFORM
        Weighting strategy to apply before optimal transport. Available
        strategies are:

            * TR_WS_UNIFORM (default), all points are weighted the same
            * TR_WS_AUTO, use a density-based weighting described in
              (Fouché, 2021)
            * TR_WS_LABELS, use a labels-balancing strategy so that for
              each label set, relative weight in both datasets is equal
            * TR_WS_CUSTOM, use weights provided by the user

    weights: (n,) np.ndarray, default = None
        User-custom weights, use weighting_strategy=TR_WS_CUSTOM to enable
        this parameter.

    labels: (n,) np.ndarray, default = None
        User-custom labels, notably used for weighting_strategy=TR_WS_LABELS.

    low_memory: bool, default = False
        Trades computation time for lower memory usage. We plan on making
        it influence more pipeline parts in the future.
    
    verbose: int, default = 1
        Defines the logging level.
        0: Disabled
        1: Informations
        2: Debug
    """
    def __init__(
            self,
            X: np.ndarray,
            metric: str = "euclidean",
            metric_kwargs: dict = {},
            geodesic: bool = False,
            normalize: bool = False,
            n_comps: int = -1,
            custom_pca: PCA = None,
            n_neighbors: int = 15,
            n_hops: int = 0,
            weighting_strategy: int = TR_WS_UNIFORM,
            weights: np.ndarray = None,
            labels: np.ndarray = None,
            low_memory: bool = False,
            verbose: bool = 0
    ):

        self.layers = { "raw": X }
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.geodesic = geodesic
        self.normalize = normalize
        self.n_comps = n_comps
        self.custom_pca = custom_pca
        self.n_neighbors = n_neighbors
        self.n_hops = n_hops
        self.weighting_strategy = weighting_strategy
        self.low_memory = low_memory
        self.verbose = verbose
        self.attributes = {}
        self.extras = {} 

        # Assert nonempty X
        self.X = self.layers["raw"] # shortcut
        assert len(self.X.shape) > 0,\
            "Cannot initialize an empty TData."

        n = len(self)

        # Weights
        if self.weighting_strategy == TR_WS_CUSTOM:
            assert weights is not None,\
                "Error: custom weighting strategy but no "\
                "weights provided."
            self.add_attribute(
                "weights", weights / weights.sum()
            )
        else:
            # By default, uniform_weights
            self.add_attribute("weights", np.ones(n, dtype=np.float64) / n)

        # Labels
        if labels is not None:
            self.add_attribute("labels", labels)

        self.validate_parameters()

        self.add_attribute("anchors", np.ones(len(self), dtype=bool))
        self.add_attribute("mapping", np.arange(len(self), dtype=int))

        # Normalizing if necessary
        if self.normalize:
            self.add_layer("raw_normalized", col_normalize(self.X))

        # Computing PCA
        if self.n_comps > 0:
            self.pca()

        # Compute neighbors in all cases for now
        if (
            self.n_hops > 0
            or self.geodesic
            or self.weighting_strategy == TR_WS_AUTO
        ):
            self.neighbors()

        # Vertex cover
        if self.n_hops > 0:
            self.vertex_cover()

        self._log(f"TData initialized, length {len(self)}", level=2)


    def validate_parameters(self):

        n = len(self)

        # Valid number of components
        assert isinstance(self.n_comps, int)\
            and self.n_comps >= -1\
            and self.n_comps <= self.X.shape[1],\
            f"Invalid number of components for PCA: {self.n_comps}."

        # Valid number of neighbors
        assert isinstance(self.n_neighbors, int)\
            and self.n_neighbors >= 1\
            and self.n_neighbors < n,\
            f"Invalid number of neigbors: {self.n_neigbors}."
        

    def __len__(self):
        return self.layers["raw"].shape[0]


    def __str__(self):
        return f"<TData> of length {len(self)}"


    def _log(self, s: str, end: str = "\n", header: bool = True, level=2) -> None:
        # Only prints for now, can later be pipelined into other streams
        if level > self.verbose:
            return
        if header:
            s = f"# Transmorph/TData > {s}"
        print(s, end=end)


    def add_attribute(self, key, value):
        assert isinstance(value, np.ndarray) and \
            value.shape[0] == len(self),\
            f"Incompatible values for attribute {key}."
        self.attributes[key] = value


    def get_attribute(self, key, subsample=False):
        values = self.attributes.get(key, None)
        if subsample and values is not None:
            anchors = self.get_attribute("anchors")
            return values[anchors]
        return values
        

    def add_layer(self, key, value):
        assert isinstance(value, np.ndarray) and \
            value.shape[0] == len(self),\
            f"Incompatible shapes: {value.shape} != {len(self)}"
        self.layers[key] = value
        

    def get_layer(self, key, subsample=False):
        values = self.layers.get(key, None)
        if subsample and values is not None:
            anchors = self.get_attribute("anchors")
            return values[anchors,:]
        return values


    def get_working_layer(self, subsample=False):
        # Returns the default layer to work with
        layer = self.get_layer("pca", subsample=subsample)
        if layer is not None:
            return layer
        layer = self.get_layer("raw_normalized", subsample=subsample)
        if layer is not None:
            return layer
        return self.get_layer("raw", subsample=subsample)
    

    def add_extra(self, key, value):
        self.extras[key] = value


    def get_extra(self, key):
        return self.extras.get(key, None)


    def pca(self):
        """
        Computes the internal PCA layer.
        """
        X = self.get_working_layer()
        if self.custom_pca is None:
            self._log("Fitting PCA...")
            pca = PCA(n_components=self.n_comps)
            pca = pca.fit(X)
        else:
            self._log("Using custom PCA.")
            pca = self.custom_pca

        self.add_layer("pca", pca.transform(self.X))
        self.add_extra("pca", pca)


    def neighbors(self, use_subsample=False):
        """
        Computes the (n,n) nearest neighbors matrix.
        """
        umap_graph = compute_umap_graph(
            self.get_working_layer(subsample=use_subsample),
            metric=self.metric,
            metric_kwargs=self.metric_kwargs,
            n_neighbors=self.n_neighbors,
            low_memory=self.low_memory
        )

        if use_subsample:

            self.add_extra("sigmas_subsampled", umap_graph["sigmas"])
            self.add_extra("rhos_subsampled", umap_graph["rhos"])
            self.add_extra( # Binary sparse matrix
                "distance_graph_subsampled",
                umap_graph["fuzzy_distances"]
            )
            self.add_extra(
                "strength_graph_subsampled",
                umap_graph["strengths"]
            )

        else:

            self.add_attribute("sigmas", umap_graph["sigmas"])
            self.add_attribute("rhos", umap_graph["rhos"])
            self.add_extra( # Binary sparse matrix
                "distance_graph",
                umap_graph["fuzzy_distances"]
            )
            self.add_extra(
                "strength_graph",
                umap_graph["strengths"]
            )
            self.add_extra(
                "knn_indices",
                umap_graph["knn_indices"]
            )


    def vertex_cover(self):
        """
        Computes the $hops-vertex cover.
        """
        edges = self.get_extra("strength_graph")
        assert edges is not None, \
            "Neighbors must be computed first."

        self._log(f"Computing vertex cover, {self.n_hops} hops.")
        anchors, mapping = vertex_cover(
            edges.indptr,
            edges.indices,
            hops=self.n_hops
        )
        self.add_attribute("anchors", anchors.astype(bool))
        self.add_attribute("mapping", mapping)
        self.neighbors(use_subsample=True) # Compute sigmas, rhos

        self._log(f"{int(anchors.sum())} anchors kept (out of {len(self)} points)")


    def smooth_by_pooling(self, layer=None):
        if layer is None:
            X = self.get_working_layer()
        else:
            X = self.get_layer(layer)

        knn_indices = self.get_extra("knn_indices")
        assert knn_indices is not None, \
            "kNN-graph must be computed prior to pooling."
        X_stabilized = within_modality_stabilize(
            X,
            knn_indices,
            n_neighbors = self.n_neighbors
        )
        self.add_layer("pooled", X_stabilized)


    def distance(self,
                 other=None,
                 metric: str = None,
                 subsampling: bool = False,
                 result_full_if_subsampling: bool = False,
                 layer: str = None):
        """
        Returns the inner pairwise distance matrix by default, or the
        (n,m) pairwise distance matrix if another TData is provided.

        Parameters:
        -----------

        other: TData
            Returns the inner pairwise distance matrix by default, or the
            (n,m) pairwise distance matrix if another TData is provided.

        metric: str or Callable
            scipy-compatible metric. 

        geodesic: bool
            Only works for other = None. Computes the distance between points
            on a graph. Nearest neighbors must have been computed first.

        subsample: bool
            Use the anchors representation instead of the full-dataset one
            for geodesic computation.

        return_full_size:
            Returns full-dataset representation.

        layer: str
            Layer to use for distance computation. Typical layers are
            "raw", "pca".
        """
        if other is None:
            other = self

        if other is self and self.geodesic:

            distance_graph = self.get_extra("distance_graph")
            assert distance_graph is not None, \
                "Neighbors must be computed first."

            if subsampling:
                anchors = self.get_attribute("anchors")
                graph_matrix = graph_matrix[anchors][:,anchors]

            Dmatrix = dijkstra(graph_matrix)
            M = Dmatrix[Dmatrix != float("inf")].max() # removing inf values
            Dmatrix[Dmatrix == float("inf")] = M

        else:

            if metric is None:
                metric = self.metric
            if layer is None:
                X, Y = (
                    self.get_working_layer(subsample=subsampling),
                    other.get_working_layer(subsample=subsampling)
                )
            else:
                X, Y = (
                    self.get_layer(layer, subsample=subsampling),
                    other.get_layer(layer, subsample=subsampling)
                )

            assert X is not None, \
                f"Unknown layer in self: {layer}"
            assert Y is not None, \
                f"Unknown layer in other: {layer}"

            X_bef = self.get_working_layer()
            anchors = self.get_attribute("anchors")
            Dmatrix = cdist(X, Y, metric=metric, **self.metric_kwargs)

        if subsampling and result_full_if_subsampling:

            mapping = self.get_attribute("mapping")
            distance_map = np.array([
                distance_graph[i, mapping[i]]
                for i in range(len(self))
            ])
            Dmatrix = fill_dmatrix(
                Dmatrix,
                anchors,
                mapping,
                distance_map
            )

        return Dmatrix


    def compute_weights(
            self,
            other_if_labels=None
    ):
        """
        Returns weights associated to self.x points. In charge of computing
        them if needed.
        """
        n = len(self)
        weights = np.zeros(len(self), dtype=np.float64)
        anchors = self.get_attribute("anchors")

        if self.weighting_strategy == TR_WS_UNIFORM:        

            n_anchors = anchors.sum()
            weights[anchors] = 1.0 / n_anchors

        if self.weighting_strategy == TR_WS_AUTO: # WOTi
            subsampling = self.n_hops > 0
            if subsampling:
                sigmas = self.get_extra("sigmas_subsampled")
                rhos = self.get_extra("sigmas_subsampled")
            else:
                sigmas = self.get_attribute("sigmas", subsample=False)
                rhos = self.get_attribute("rhos", subsample=False)
            assert (
                sigmas is not None and
                rhos is not None
            ), "Sigmas and rhos must be computed for WOTi strategy."

            Dmatrix = self.distance(
                metric="euclidean", # Euclidean is fixed here
                subsampling=subsampling,
                result_full_if_subsampling=False
            )
            self._log("Solving the QP to find weights...")
            result_weights = normal_kernel_weights(
                Dmatrix,
                scales=sigmas,
                offsets=rhos,
                alpha_qp=1.0
            )
            anchors = self.get_attribute("anchors")
            weights[anchors] = result_weights

        elif self.weighting_strategy == TR_WS_LABELS:

            assert other_if_labels is not None, \
                "Missing labels for reference dataset."
            subsampling = self.n_hops > 0
            self_labels = self.get_attribute(
                "labels",
                subsample=subsampling
            )
            other_labels = other_if_labels.get_attribute(
                "labels",
                subsample=subsampling
            )
            assert (
                self_labels is not None
                and other_labels is not None
            ), "Missing labels in self or others."
            self_weights, other_weights = weight_per_label(
                self_labels,
                other_labels
            )

            self_anchors, other_anchors = (
                self.get_attribute("anchors"),
                other_if_labels.get_attribute("anchors")
            )
            other_weights_full = np.zeros(len(other_if_labels), dtype=np.float64)
            weights[self_anchors] = self_weights
            other_weights_full[other_anchors] = other_weights
            other_weights_full /= other_weights_full.sum()
            other_if_labels.add_attribute("weights", other_weights_full)

        weights /= weights.sum()
        self.add_attribute("weights", weights)


    def get_barycenter(self, subsample=True):
        """
        Returns the weighted dataset barycenter.
        """
        return np.diag(self.weights() @ self.X[self.anchors]).sum(axis=0)
