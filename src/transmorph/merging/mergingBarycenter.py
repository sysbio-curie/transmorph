#!/usr/bin/env python3

import warnings
import numpy as np

from scipy.sparse import csr_matrix

from .mergingABC import MergingABC

from ..matching.matchingABC import MatchingABC


class MergingBarycenter(MergingABC):
    """
    Barycentric merging is the most naive way of merging datasets.
    Given n vectorized datasets x1, ..., xn and a reference dataset y,
    and n row-normalized matchings m1, ..., mn, we compute the following
    projection:

        f(xi) = mi @ y

    In other terms, every sample from xi is projected to the weighted sum
    of its matches in y. This works quite well with optimal transport-based
    methods, but is quite limited in the MNN case due to unmatched samples.

    Merging type
    ------------
    np.ndarray containing an embedding of all datasets in y's space.
    """

    def __init__(self, matching: MatchingABC):
        MergingABC.__init__(self, matching)

    def _check_input(self, matching: MatchingABC) -> None:
        """
        Raises an additional warning if some source samples are unmatched.
        """
        assert matching.use_reference and matching.reference is not None, (
            "Error: Matching must be fit with reference=X_target for "
            "barycentric merging."
        )
        for k in range(matching.n_datasets):
            T = matching.get_matching(k, normalize=True)
            if any(T.sum(axis=1) == 0.0):
                warnings.warn(
                    "Warning: Some samples are unmatched and will result on origin. "
                    "You may want to try another matching (e.g. MatchingEMD or"
                    "MatchingGW) or merging (e.g. MergingLinearCorrection) strategy."
                )

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
            projection = T @ reference
            output[offset : offset + n] = projection
            offset += n
        return output
