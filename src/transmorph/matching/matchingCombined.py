#!/usr/bin/env python3

from .matchingABC import MatchingABC
from typing import List

from scipy.sparse import csr_matrix

from anndata import AnnData


class MatchingCombined(MatchingABC):
    """
    Combines several matchings following a chosen aggregation strategy,
    among:
    - additive, where the matching between two points is the arithmetical
      average over all (normalized) matchings, then normalized
    - multiplicative, where the matching between two points is the product
      over all (normalized) matchings, then normalized
    - minimum, where the matching between two points is the minimum over
      all (normalized) matchings, then normalized
    - maximum, where the matching between two points is the maximum over
      all (normalized) matchings, then normalized

    Parameters
    ----------
    matchings: List[MatchingABC]
        List of source matchings to aggregate

    mode: str, default = "additive"
        String identifier in ("additive", "multiplicative", "minimum",
        "maximum") describing chosen strategy.
    """

    def __init__(
        self,
        matchings: List[MatchingABC],
        mode: str = "additive",
    ):
        super().__init__(metadata_keys=[])
        self.source_matchings = matchings
        assert mode in (
            "additive",
            "multiplicative",
            "minimum",
            "maximum",
        ), f"Unrecognized mode: {mode}"
        self.mode = mode

    def normalize(self, T_matching: csr_matrix) -> csr_matrix:
        """
        Helper function to row-normalize a matching matrix.
        """
        T_matching = csr_matrix(T_matching / T_matching.sum(axis=1))
        return csr_matrix(T_matching + T_matching.T - T_matching.multiply(T_matching))

    def _match2(self, adata1: AnnData, adata2: AnnData) -> csr_matrix:
        """
        Combines results from self.matchings between adata1 and adata2.
        """
        assert all(
            m.fitted for m in self.source_matchings
        ), "All matchings must be fitted."
        Ts = []
        for matching in self.source_matchings:
            Ti = self.normalize(matching.get_matching(adata1, adata2))
            Ts.append(Ti)
        T = Ts[0]
        for Ti in Ts[1:]:
            if self.mode == "additive":
                T += Ti
            elif self.mode == "multiplicative":
                T = T.multiply(Ti)
            elif self.mode == "minimum":
                T = T.minimum(Ti)
            elif self.mode == "maximum":
                T = T.maximum(Ti)
        return self.normalize(T)
