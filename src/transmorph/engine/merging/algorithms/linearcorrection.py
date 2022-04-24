#!/usr/bin/env python3

import numpy as np

from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from typing import List

from ..merging import Merging
from ...traits.usesneighbors import UsesNeighbors
from ...traits.usesreference import UsesReference


class LinearCorrection(Merging, UsesNeighbors, UsesReference):
    """
    LinearCorrection is a way to merge vectorized datasets embedded
    in the same vector space onto a reference, aiming to solve issues
    of barycentric merging with partial matchings and overfitting.

    Starting from two datasets X (source) and Y (reference) and a
    matching M[x, y] > 0 if x and y are matched, M[x, y] = 0 otherwise,
    we compute correction vectors c(x) between matched points and the
    barycenter of their matches,

        c(x) = (1 / \\sum_y M[x, y]) * (\\sum_y M[x, y] * y) - x

    We end up with a set of correction vectors c(x) for some x, and
    need to guess c(x) for unmatched x.

    The strategy we adopt is to build a k-NN graph of X points,
    smoothing correction vectors with respect to kNN neighborhood.
    Doing so, all points that are matched or neighboring a matched
    point are associated with a smoothed correction vector. To
    finish, we set for all uncorrected points a correction vector
    equal to the one of their closest corrected neighbor.
    """

    def __init__(self):
        Merging.__init__(
            self,
            preserves_space=True,
            str_identifier="LINEAR_CORRECTION",
            matching_mode="normalized",
        )
        UsesNeighbors.__init__(self)
        UsesReference.__init__(self)

    def project(
        self,
        X_src: np.ndarray,
        X_ref: np.ndarray,
        k_src: int,
        T: csr_matrix,
    ) -> np.ndarray:
        """
        Returns the projected view of X onto Y given the matching T
        """
        from ...._settings import settings

        ref_locations = T @ X_ref
        corr_vectors = ref_locations - X_src

        # We smooth correction vectors wrt neighborhood
        n = X_src.shape[0]
        knn_graph = self.get_neighbors_graph(k_src, mode="edges")
        matched = ref_locations.any(axis=1)
        knn_graph += np.diag(matched)
        n_connections = np.array(knn_graph[:, matched].sum(axis=1)).reshape(-1)
        connected = n_connections > 0
        corr_vectors_smooth = np.zeros(X_src.shape)
        corr_vectors_smooth[connected] = (
            np.diag(1.0 / n_connections[connected])
            @ knn_graph[connected][:, matched]
            @ corr_vectors[matched]
        )
        # Correct the remaining points
        if not np.all(connected):
            matched = X_src[connected]
            unconnected = np.logical_not(connected)
            unmatched = X_src[unconnected]
            dists = cdist(
                unmatched,
                matched,
                metric=settings.neighbors_metric,
                **settings.neighbors_metric_kwargs
            )
            idx_ref = np.arange(n)[connected][np.argmin(dists, axis=1)]
            corr_vectors_smooth[unconnected] = corr_vectors_smooth[idx_ref]
        return X_src + corr_vectors_smooth

    def transform(self, datasets: List[np.ndarray]) -> List[np.ndarray]:
        """
        Computes correction vectors, then transforms.
        """
        k_ref = self.reference_index
        assert k_ref is not None, "No reference provided."
        X_ref = self.get_reference_item(datasets)
        assert X_ref is not None, "No reference provided."
        result = []
        for k, X in enumerate(datasets):
            if X is X_ref:
                result.append(X_ref)
                continue
            T = self.get_matching(k, k_ref)
            projection = self.project(X, X_ref, k, T)
            result.append(projection)
        return result
