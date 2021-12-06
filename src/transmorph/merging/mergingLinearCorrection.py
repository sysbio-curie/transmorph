#!/usr/bin/env python3

import numpy as np

from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix

from .mergingABC import MergingABC
from ..matching.matchingABC import MatchingABC
from ..utils import nearest_neighbors


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
        matching: MatchingABC,
        n_neighbors: int = 5,
        metric: str = "sqeuclidean",
        metric_kwargs: dict = {},
        use_nndescent: bool = False,
        low_memory: bool = False,
        n_jobs: int = -1,
    ):
        MergingABC.__init__(self, matching)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.use_nndescent = use_nndescent
        self.low_memory = low_memory
        self.n_jobs = n_jobs

    def _check_input(self) -> None:
        super()._check_input()
        matching = self.matching
        reference = matching.get_reference()
        assert matching.use_reference and reference is not None, (
            "Error: Matching must be fit with reference=X_target for "
            "barycentric merging."
        )
        assert all(
            dataset.shape[1] == reference.shape[1] for dataset in matching.datasets
        ), (
            "Error: Cannot use"
            " LinearCorrection to merge datasets from different"
            " spaces. Try Barycenter or MDI instead."
        )

    def _project(self, X, Y, T):
        ref_locations = np.asarray(T @ Y)
        corr_vectors = ref_locations - X
        n_neighbors = self.n_neighbors
        if n_neighbors == 0:  # Equiv. to barycentric
            return X + corr_vectors

        # We smooth correction vectors wrt neighborhood
        n = X.shape[0]
        matched = ref_locations.any(axis=1)
        knn_graph = nearest_neighbors(  # Neighborhood matrix
            X,
            metric=self.metric,
            metric_kwargs=self.metric_kwargs,
            symmetrize=True,
            use_nndescent=self.use_nndescent,
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
        return X + corr_vectors_smooth

    def transform(self) -> np.ndarray:
        matching = self.matching
        reference = matching.get_reference()
        assert reference is not None
        output = np.zeros(
            (
                reference.shape[0]
                + sum(dataset.shape[0] for dataset in matching.datasets),
                reference.shape[1],
            )
        )
        output[: reference.shape[0]] = reference
        offset = reference.shape[0]
        for k, dataset in enumerate(matching.datasets):
            n = dataset.shape[0]
            T = matching.get_matching(k, normalize=True)
            if type(T) is csr_matrix:
                T = T.toarray()
            projection = self._project(dataset, reference, T)
            output[offset : offset + n] = projection
            offset += n
        return output
