#!/usr/bin/env python3

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import dijkstra

from .constants import *
from .density import sigma_search
from .density import normal_kernel_weights
from .utils import col_normalize
from .utils import fill_dmatrix
from .utils import symmetrize
from .utils import vertex_cover
from .utils import weight_per_label


class TData:
    """
    Custom data object used by the main module. Features
    weight computation, subsampling, custom distance, labels handling...

    Parameters:
    -----------

    X: (n,d) np.ndarray to embed.

    weights: (n,) np.ndarray
        Weights vector, default: uniform

    labels: (n,) np.ndarray
        Labels vector for supervised problem.

    normalize: bool, default = False
        Wether to use a column-normalized representation to compute
        PCA, distance etc.
    
    verbose: int, default = 1
        Defines the logging level.
        0: Disabled
        1: Informations
        2: Debug
    """
    def __init__(self,
                 X: np.ndarray,
                 weights: np.ndarray = None,
                 labels: np.ndarray = None,
                 normalize: bool = False,
                 verbose: bool = 0):

        self.layers = {'raw': X}
        self._weights = weights # Acess via weights()
        self._labels = labels
        self.normalize = normalize
        self.verbose = verbose

        self.validate_parameters()

        self.X = self.layers['raw'] # shortcut
        self._neighbors = None

        # [T, T, F, F, ..., T] representers r
        # Default: all points
        self.anchors = np.ones(len(self)).astype(bool)

        # [ 0, 1, 1, 5, ..., n-1 ] mapping i -> r_i
        # Default: each point is its anchor
        self.anchors_map = np.arange(len(self))
        
        # [.25, .37, ...] mapping i -> d(i, r_i)
        # Default: each point is at distance 0 from its anchor
        self.distances_map = np.zeros(len(self)).astype(np.float32)

        self._log("TData initialized, length %i" % len(self), level=3)


    def validate_parameters(self):

        assert len(self.layers['raw'].shape) > 0, \
            "Cannot initialize an empty TData."
        n = len(self)

        assert self._weights is None or len(self._weights) == n,\
            "Inconsistent size between _weights and dataset. Expected %i,\
             found %i." % (n, len(self._weights))

        # By default, uniform _weights
        if self._weights is None:
            self._weights = np.ones(n) / n
        else:
            self._weights /= self._weights.sum()

        assert self._labels is None or len(self._labels) == n,\
            "Inconsistent size between labels and dataset. Expected %i,\
             found %i." % (n, len(self._labels))


    def __len__(self):
        return self.layers['raw'].shape[0]


    def __str__(self):
        return "<TData> of length %i" % len(self)


    def _log(self, s: str, end: str = '\n', header: bool = True, level=2) -> None:
        # Only prints for now, can later be pipelined into other streams
        if level > self.verbose:
            return
        if header:
            s = "(Transmorph/TData) > %s" % s
        print(s, end=end)


    def weights(self):
        w = self._weights[self.anchors]
        return w / w.sum()


    def labels(self):
        return self._labels[self.anchors]

    
    def pca(self,
            n_components: int = 15,
            subsample: bool = False,
            other = None):
        """
        Computes the internal PCA layer.
        subsample = True -> use vertex cover instead of whole dataset
        """

        pca = PCA(n_components=n_components)

        assert n_components <= self.X.shape[1], \
            "n_comps must be lesser or equal to data dimension."

        if other is None:
            X = self.X
            if subsample:
                X = X[self.anchors]
            if self.normalize:
                X = col_normalize(X)
            pca = pca.fit(X)
            self.layers['pca'] = pca.transform(self.X)
        else: # Shared PCA view between two datasets
            assert self.X.shape[1] == other.X.shape[1], \
                "Incompatible dimensions for distance computation: %i != %i" \
                % (self.X.shape[1], other.X.shape[1])
            X, Y = self.X, other.X
            if subsample:
                X, Y = X[self.anchors], Y[other.anchors]
            if self.normalize:
                X = col_normalize(X)
            if other.normalize:
                Y = col_normalize(Y)

            pca = pca.fit(
                np.concatenate( (X, Y), axis=0 )
            )
            Xtot = pca.transform(
                np.concatenate( (self.X, other.X), axis=0 )
            )
            self.layers['pca'] = Xtot[:len(self)]
            other.layers['pca'] = Xtot[len(self):]


    def neighbors(self,
                  n_neighbors=5,
                  metric='sqeuclidean',
                  layer='raw',
                  self_edit=True,
                  subsample=False):
        """
        Computes the (n,n) nearest neighbors matrix.
        """
        assert layer in self.layers, \
            "Unrecognized layer: %s" % layer
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        X = self.layers[layer]
        if subsample:
            X = X[self.anchors]
        if self.normalize and layer == 'raw':
            X = col_normalize(X)
        nn.fit(X)
        nngraph = symmetrize(
            nn.kneighbors_graph(mode='distance')
        )
        if self_edit:
            self._neighbors = nngraph
        return nngraph


    def select_representers(self, hops=2):
        """
        Computes the $hops-vertex cover.
        """
        assert self._neighbors is not None, \
            "Neighbors must be computed first."

        self.anchors, self.anchors_map = vertex_cover(
            self._neighbors.indptr,
            self._neighbors.indices,
            hops=hops
        )
        self.anchors = self.anchors.astype(bool)
        self.distances_map = np.array([
            self._neighbors[i, j]
            for (i, j) in enumerate(self.anchors_map)
        ])


    def distance(self,
                 other=None,
                 metric: str = 'sqeuclidean',
                 geodesic: bool = False,
                 subsample: bool = False,
                 return_full_size: bool = True,
                 layer: str = 'raw'):
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
            'raw', 'pca'.
        """
        assert layer in self.layers, \
            "Unrecognized layer: %s" % layer

        if geodesic:
            assert other is None, \
                "Unable to compute geodesic distance between datasets."

            assert self._neighbors is not None, \
                "Neighbors must be computed first."

            graph_matrix = self._neighbors
            if subsample:
                graph_matrix = graph_matrix[self.anchors][:,self.anchors]
            Dmatrix = dijkstra(graph_matrix)
            M = Dmatrix[Dmatrix != float('inf')].max() # removing inf values
            Dmatrix[Dmatrix == float('inf')] = M
            if subsample and return_full_size:
                Dmatrix = fill_dmatrix(Dmatrix,
                                       self.anchors,
                                       self.anchors_map,
                                       self.distances_map)
            return Dmatrix

        # Cacheing the inner pairwise matrix if needed
        if other is None:
            other = self
        else:
            assert layer in other.layers, \
                "Unrecognized layer: %s" % layer
            assert self.layers[layer].shape[1] == other.layers[layer].shape[1], \
                "Incompatible dimensions for distance computation: %i != %i" \
                % (self.layers[layer].shape[1], other.layers[layer].shape[1])

        X, Y = self.layers[layer], other.layers[layer]
        if subsample:
            X, Y = X[self.anchors], Y[other.anchors]
        if self.normalize and layer == 'raw':
            X = col_normalize(X)
        if other.normalize and layer == 'raw':
            Y = col_normalize(Y)
        Dmatrix = cdist(X, Y, metric=metric)
        if subsample and return_full_size:
            Dmatrix = fill_dmatrix(
                Dmatrix,
                self.anchors,
                self.anchors_map,
                self.distances_map
            )
        return Dmatrix


    def compute_weights(self,
                        method=TR_WS_AUTO,
                        layer='raw',
                        subsample=False,
                        other=None):
        """
        Returns weights associated to self.x points. In charge of computing
        them if needed.
        """
        if method == TR_WS_UNIFORM:
            self._weights = np.ones(len(self)) / len(self)
        if method == TR_WS_AUTO:
            self._log("Searching for sigma...", end=' ')
            Dmatrix = self.distance(
                metric="euclidean",
                layer=layer,
                subsample=subsample,
                return_full_size=False
            )
            scale = sigma_search(Dmatrix)
            self._log("Found: %f" % scale, header=False)
            self._log("Solving the QP to find weights...", end=' ')
            weights = normal_kernel_weights(
                Dmatrix,
                scale=scale,
                alpha_qp=1.0)
            if subsample:
                self._weights = np.zeros(len(self))
            self._weights[self.anchors] = weights
            self._log("Done.", header=False)
        if method == TR_WS_LABELS:
            assert other is not None, "Missing labels for reference dataset."
            self._weights, other._weights = weight_per_label(self._labels, other._labels)


    def get_barycenter(self, subsample=True):
        """
        Returns the weighted dataset barycenter.
        """
        return np.diag(self.weights() @ self.X[self.anchors]).sum(axis=0)
