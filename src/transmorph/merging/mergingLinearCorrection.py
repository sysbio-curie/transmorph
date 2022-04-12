#!/usr/bin/env python3

import numpy as np
import warnings

from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, csc_matrix
from anndata import AnnData
from typing import List, Optional, Union

from transmorph.utils.anndata_interface import get_matrix

from .mergingABC import MergingABC
from ..matching.matchingABC import MatchingABC
from ..utils import nearest_neighbors, pca


class MergingLinearCorrection(MergingABC):
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

    Parameters
    ----------
    matching: MatchingABC
        Fitted, referenced matching between datasets.

    n_neighbors: int, default = 5
        Number of neighbors to use to compute correction smoothing.

    metric: str, default = "sqeuclidean"
        Distance metric to use for kNN graph construction.

    metric_kwargs: dict, default = {}
        Extra parameters for the metric to pass to scipy.distance.cdist.

    use_nndescent: bool, default = False
        Use the quicker, approximate solver for nearest neighbors.

    low_memory: bool, default = False
        If using use_nndescent=True, switch to a low memory/high time
        strategy.

    n_jobs: int, default = -1
        Number of threads to use.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        learning_rate: float = 1.0,
        n_neighbors: int = 5,
        metric: str = "sqeuclidean",
        metric_kwargs: dict = {},
        jitter: Optional[float] = None,
        low_memory: bool = False,
        n_jobs: int = -1,
    ):
        super().__init__(use_reference=True)
        self.n_pcs = n_components
        assert (
            learning_rate > 0.0 and learning_rate <= 1.0
        ), f"Learning rate {learning_rate} out of bounds (0.0, 1.0]."
        self.lr = learning_rate
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        if jitter is None:
            jitter = 0.0 if self.lr < 1.0 else 0.01
        if jitter > 0 and self.lr < 1.0:
            warnings.warn("Jitter > 0 and lr < 1.0 is discouraged.")
        self.jitter = jitter
        self.low_memory = low_memory
        self.n_jobs = n_jobs

    def do_jitter(self, X):
        stdev = self.jitter * (np.max(X, axis=0) - np.min(X, axis=0))
        return X + np.random.randn(*X.shape) * stdev

    def project(self, X, Y, T):
        """
        Returns the projected view of X onto Y given the matching T
        """
        ref_locations = np.asarray(T @ Y)
        corr_vectors = ref_locations - X
        n_neighbors = self.n_neighbors
        if n_neighbors == 0:  # Equiv. to barycentric
            return X + corr_vectors

        # We smooth correction vectors wrt neighborhood
        n = X.shape[0]
        X_knn = X
        if self.n_pcs is not None:
            X_knn = pca(X, n_components=self.n_pcs)
        elif X.shape[1] > 100:
            warnings.warn(
                "High dimensional data detected. Please consider setting "
                "'n_components' to a value in order to build the kNN graph "
                "based on a PC view."
            )
        matched = ref_locations.any(axis=1)
        knn_graph = nearest_neighbors(  # Neighborhood matrix
            X_knn,
            metric=self.metric,
            metric_kwargs=self.metric_kwargs,
            symmetrize=True,
            n_neighbors=n_neighbors,
            low_memory=self.low_memory,
            n_jobs=self.n_jobs,
        ).astype(bool)
        knn_graph += np.diag(matched)
        n_connections = np.asarray(knn_graph[:, matched].sum(axis=1)).reshape(
            n,
        )
        connected = n_connections > 0
        corr_vectors_smooth = np.zeros(X.shape, dtype=np.float32)
        corr_vectors_smooth[connected] = (
            np.diag(1.0 / n_connections[connected])
            @ knn_graph[connected][:, matched]
            @ corr_vectors[matched]
        )
        # Correct the remaining points
        if not np.all(connected):
            unconnected = np.logical_not(connected)
            unmatched = X[unconnected]
            matched = X[connected]
            dists = cdist(unmatched, matched, metric=self.metric, **self.metric_kwargs)
            idx_ref = np.arange(n)[connected][np.argmin(dists, axis=1)]
            corr_vectors_smooth[unconnected] = corr_vectors_smooth[idx_ref]
        return X + corr_vectors_smooth * self.lr

    def fit(
        self,
        datasets: List[AnnData],
        matching: Optional[MatchingABC] = None,
        matching_mtx: Optional[Union[csr_matrix, np.ndarray]] = None,
        X_kw: str = "",
        reference_idx: int = -1,
    ) -> List[np.ndarray]:
        self._check_input(datasets, matching, matching_mtx, X_kw, reference_idx)
        adata_ref = datasets[reference_idx]
        list_mtx = [get_matrix(adata, X_kw) for adata in datasets]
        assert adata_ref is not None
        result = []
        for k, adata in enumerate(datasets):
            if k == reference_idx:
                result.append(list_mtx[k])
                continue
            if matching is not None:
                T = matching.get_matching(adata, adata_ref, row_normalize=True)
            elif matching_mtx is not None:  # Works as an elif
                T = matching_mtx
                if T.shape[0] == adata_ref.n_obs:
                    T = T.T
                if type(T) is csc_matrix or type(T) is csr_matrix:
                    T = T.toarray()
                assert type(T) is np.ndarray, f"Unrecognized type: {type(T)}"
                T = csr_matrix(T / T.sum(axis=1, keepdims=True))
            else:
                raise AssertionError("matching or matching_mtx must be set.")
            assert type(T) is csr_matrix, f"Unrecognized type: {type(T)}"
            if type(T) is csr_matrix:
                T = T.toarray()
            projection = self.project(list_mtx[k], list_mtx[reference_idx], T)
            result.append(projection)

        if self.jitter:
            result = [self.do_jitter(X) for X in result]
        return result
