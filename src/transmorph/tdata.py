#!/usr/bin/env python3

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

from .constants import *
from .density import sigma_search
from .density import normal_kernel_weights
from .utils import col_normalize
from .utils import symmetrize
from .utils import vertex_cover
from .utils import weight_per_label

class TData:
    """
    # TODO: write doctring
    # TODO: slicing
    """
    def __init__(self,
                 X: np.ndarray,
                 is_weighted: bool = True,
                 weights: np.ndarray = None,
                 labels: np.ndarray = None,
                 normalize: bool = False,
                 verbose: bool = 0):

        self.layers = {'raw': X}
        self.weights = weights
        self.labels = labels
        self.normalize = normalize
        self.verbose = verbose

        self.validate_parameters()

        self.X = self.layers['raw'] # shortcut
        self._neighbors = None
        self.anchors = np.arange(len(self))
        self.anchors_map = np.arange(len(self))

        self._log("TData initialized, length %i" % len(self), level=3)


    def validate_parameters(self):

        n = len(self)
        assert self.weights is None or len(self.weights) == n,\
            "Inconsistent size between weights and dataset. Expected %i,\
             found %i." % (n, len(self.weights))

        if self.weights is None:
            self.weights = np.ones(n) / n
        else:
            self.weights /= self.weights.sum()

        assert self.labels is None or len(self.labels) == n,\
            "Inconsistent size between labels and dataset. Expected %i,\
             found %i." % (n, len(self.labels))


    def __len__(self):
        return len(self.layers['raw'])


    def __str__(self):
        return "<TData> of length %i" % len(self)


    def _log(self, s: str, end: str = '\n', header: bool = True, level=2) -> None:
        # Only prints for now, can later be pipelined into other streams
        if level > self.verbose:
            return
        if header:
            s = "(Transmorph/TData) > %s" % s
        print(s, end=end)


    def pca(self,
            n_components: int = 15,
            other = None):
        pca = PCA(n_components=n_components)
        if other is None:
            if self.normalize:
                self.layers['pca'] = pca.fit_transform(col_normalize(self.X))
            else:
                self.layers['pca'] = pca.fit_transform(self.X)
        else:
            assert self.X.shape[1] == other.X.shape[1], \
                "Incompatible dimensions for distance computation: %i != %i" \
                % (self.X.shape[1], other.X.shape[1])
            X, Y = self.X, other.X
            if self.normalize:
                X = col_normalize(X)
            if other.normalize:
                Y = col_normalize(Y)

            Xtot = pca.fit_transform(
                np.concatenate( (X, Y), axis=0 ))
            self.layers['pca'] = Xtot[:len(self)]
            other.layers['pca'] = Xtot[len(self):]


    def neighbors(self, n_neighbors=5, metric='sqeuclidean', layer='raw'):
        assert layer in self.layers, \
            "Unrecognized layer: %s" % layer
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        if self.normalize and layer == 'raw':
            nn.fit(col_normalize(self.layers[layer]))
        else:
            nn.fit(self.layers[layer])
        self._neighbors = nn.kneighbors_graph(mode='distance')
        symmetrize(self._neighbors)


    def select_representers(self, hops=2):
        self.anchors_map, self.anchors = vertex_cover(
            self._neighbors.indptr,
            self._neighbors.indices,
            hops=hops
        )


    def distance(self, other=None, metric='euclidean', layer='raw'):
        """
        Reuturns the inner pairwise distance matrix by default, or the
        (n,m) pairwise distance matrix if another TData is provided.

        Priority is red > nrm > raw
        """
        assert layer in self.layers, \
            "Unrecognized layer: %s" % layer

        # Cacheing the inner pairwise matrix if needed
        if other is None:
            other = self
        else:
            assert layer in other.layers, \
                "Unrecognized layer: %s" % layer
            assert self.layers[layer].shape[1] == other.layers[layer].shape[1], \
                "Incompatible dimensions for distance computation: %i != %i" \
                % (self.layers[layer].shape[1], other.layers[layer].shape[1])

        if self.normalize and layer == 'raw':
            return cdist(
                col_normalize(self.layers[layer]),
                col_normalize(other.layers[layer]),
                metric=metric)
        return cdist(self.layers[layer], other.layers[layer], metric=metric)


    def compute_weights(self, method=TR_WS_AUTO, layer='raw', other=None):
        """
        Returns weights associated to self.x points. In charge of computing
        them if needed.
        """
        if method == TR_WS_UNIFORM:
            self.weights = np.ones(len(self)) / len(self)
        if method == TR_WS_AUTO:
            self._log("Searching for sigma...", end=' ')
            scale = sigma_search(self.distance(metric="euclidean", layer=layer))
            self._log("Found: %f" % scale, header=False)
            self._log("Solving the QP to find weights...", end=' ')
            self._weights = normal_kernel_weights(
                self.distance(metric="euclidean", layer=layer),
                scale=scale,
                alpha_qp=1.0)
            self._log("Done.", header=False)
        if method == TR_WS_LABELS:
            assert other is not None, "Missing labels for reference dataset."
            self.weights, other.weights = weight_per_label(self.labels, other.labels)

    def get_barycenter(self):
        """

        """
        return (np.diag(self.weights()) @ self.x_raw).sum(axis=0)
