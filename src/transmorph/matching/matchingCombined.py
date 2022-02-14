#!/usr/bin/env python3

from .matchingABC import MatchingABC
from typing import List

from scipy.sparse import csr_matrix

from transmorph.TData import TData


class MatchingCombined(MatchingABC):
    """
    TODO
    """

    def __init__(
        self,
        matchings: List[MatchingABC],
        mode: str = "additive",
        use_sparse: bool = True,
    ):
        MatchingABC.__init__(self, use_sparse=use_sparse)
        self.mode = mode
        assert all(m.fitted for m in matchings), "All matchings must be fitted."
        self.datasets = matchings[0].datasets.copy()
        n_matchings = len(matchings[0].matchings)
        self.matchings = []
        assert all(
            len(m.matchings) == n_matchings for m in matchings
        ), "Inconsistent number of matchings."
        for matching_idx in range(n_matchings):
            Ts = [m.matchings[matching_idx] for m in matchings]
            reference_shape = Ts[0].shape
            assert all(T.shape == reference_shape for T in Ts)
            Ts = [self.normalize(T) for T in Ts]
            T_combined = Ts[0]
            if self.mode == "additive":
                for T in Ts[1:]:
                    T_combined += T
            if self.mode == "multiplicative":
                for T in Ts[1:]:
                    T_combined = T_combined.multiply(T)
            if self.mode == "intersection":
                for T in Ts[1:]:
                    T_combined = T_combined.minimum(T)
            T_combined = self.normalize(T_combined)
            self.matchings.append(T_combined)
        self.fitted = True

    def normalize(self, T_matching):
        """
        TODO
        """
        T_matching = csr_matrix(T_matching / T_matching.sum(axis=1))
        return T_matching + T_matching.T - T_matching.multiply(T_matching)

    def fit(self, datasets, reference):
        """ """
        raise NotImplementedError

    def _match2(self, t1: TData, t2: TData):
        """ """
        raise NotImplementedError
