#!/usr/bin/env python3

import numpy as np

from scipy.sparse import csr_matrix, diags
from typing import Dict, Literal, Optional

from ..matching import _TypeMatchingSet

_TypeMatchingModes = Literal["raw", "bool", "normalized"]


class UsesMatching:
    """
    Allows an object to use matching matrices between datasets.
    """

    def __init__(self, mode: _TypeMatchingModes = "raw"):
        # Read-only
        self.matchings: Optional[_TypeMatchingSet] = None
        assert mode in ("raw", "bool", "normalized")
        self.mode = mode

    def set_matchings(self, matchings: _TypeMatchingSet) -> None:
        """
        Loads the object with a read-only dictionary of matching matrices,
        coming from a LayerMatching.
        """
        assert isinstance(matchings, Dict)
        self.matchings = matchings

    def get_matching(self, idx1: int, idx2: int) -> csr_matrix:
        """
        Returns matching between datasets idx1 and idx2. Raises an exception
        if matching has not been found.
        """
        assert self.matchings is not None, "No matching provided."
        result = self.matchings.get((idx1, idx2), None)
        if result is None:
            result = self.matchings.get((idx2, idx2), None)
            assert result is not None, f"{idx1}, {idx2} not found in matching."
            result = csr_matrix(result.T)
        if self.mode == "raw":
            return result
        if self.mode == "bool":
            return result.astype(bool)
        if self.mode == "normalized":
            norm = np.array(result.sum(axis=1)).reshape(-1)
            norm[norm == 0.0] = 1.0
            return diags(1.0 / norm) @ result
        raise ValueError(f"Unknown mode {self.mode}")
