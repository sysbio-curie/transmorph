#!/usr/bin/env python3

import numpy as np

from anndata import AnnData
from typing import List, Literal, Optional, Tuple, Union

from ...utils.anndata_manager import (
    _TypePairwiseSlice,
    _TypeTotalSlice,
    get_pairwise_feature_slices,
    get_total_feature_slices,
)

_TypeAnnDataCommonModes = Literal["pairwise", "total"]


class UsesCommonFeatures:
    """
    This trait will allow an object to retrieve feature names from an AnnData
    object. They will then be used to slice count matrices in order to select
    pairwise or total common genes intersection.
    """

    def __init__(self, mode: _TypeAnnDataCommonModes):
        assert mode in ("pairwise", "total")
        self.mode = mode
        # TFS is used for mode "total", PFS for mode "pairwise"
        self.total_feature_slices: _TypeTotalSlice = []
        self.pairwise_feature_slices: _TypePairwiseSlice = {}
        self.fitted = False
        self.is_feature_space = False

    def retrieve_common_features(
        self, datasets: List[AnnData], is_feature_space: bool
    ) -> None:
        """
        Stores gene names for later use.
        """
        self.is_feature_space = is_feature_space
        if not is_feature_space:
            return
        if self.mode == "pairwise":
            self.pairwise_feature_slices = get_pairwise_feature_slices(datasets)
        elif self.mode == "total":
            self.total_feature_slices = get_total_feature_slices(datasets)
        else:
            raise ValueError(f"Unknown mode {self.mode}.")
        self.fitted = True

    def get_common_features(
        self, idx_1: int, idx_2: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a tuple containing feature slices to use between two datasets
        identified as their index. Raises a ValueError if idx_1, idx_2 is unknown.
        """
        assert self.fitted, "UsesCommonFeatures trait has not retrieved features."
        if self.mode == "pairwise":
            slices = self.pairwise_feature_slices.get((idx_1, idx_2), None)
            if slices is None:
                raise ValueError(f"No feature slice found for {idx_1}, {idx_2}.")
        elif self.mode == "total":
            assert idx_1 < len(self.total_feature_slices), f"{idx_1} out of bounds."
            assert idx_2 < len(self.total_feature_slices), f"{idx_2} out of bounds."
            slices = self.total_feature_slices[idx_1], self.total_feature_slices[idx_2]
        else:
            raise ValueError(f"Unknown mode {self.mode}.")
        return slices

    def slice_features(
        self,
        X1: np.ndarray,
        idx_1: int,
        X2: Optional[np.ndarray] = None,
        idx_2: Optional[int] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Returns a sliced view of datasets X1 and X2, indexed by idx_1 and idx_2. Raises
        a ValueError if indices are not found, or if slice size does not coincidate.
        """
        assert not ((X2 is None) ^ (idx_2 is None))
        # If we are not in feature space, we must skip the processing.
        if not self.is_feature_space:
            if X2 is None:
                return X1
            return X1, X2
        if X2 is None:
            assert (
                self.mode == "total"
            ), "Calling slice_features with one dataset is only"
            " valid for mode == 'total'."
            return X1[:, self.total_feature_slices[idx_1]]
        s1, s2 = self.get_common_features(idx_1, idx_2)
        assert s1.shape[0] == X1.shape[1], (
            f"Unexpected matrix features number. Expected {s1.shape[0]}, "
            f"found {X1.shape[1]}."
        )
        assert s2.shape[0] == X2.shape[1], (
            f"Unexpected matrix features number. Expected {s2.shape[0]}, "
            f"found {X2.shape[1]}."
        )
        return X1[:, s1], X2[:, s2]
