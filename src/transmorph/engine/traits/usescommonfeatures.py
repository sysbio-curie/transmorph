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

_TypeAnnDataCommonModes = Literal["pairwise", "total", "ignore"]


class UsesCommonFeatures:
    """
    This trait will allow an object to retrieve feature names from an AnnData
    object. They will then be used to slice count matrices in order to select
    pairwise or total common genes intersection.
    """

    def __init__(self, mode: _TypeAnnDataCommonModes):
        assert mode in ("pairwise", "total", "ignore")
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
        if not is_feature_space or self.mode == "ignore":
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
        if not self.is_feature_space or not self.fitted or self.mode == "ignore":
            if X2 is None:
                return X1
            return X1, X2
        if X2 is None:
            assert (
                self.mode == "total"
            ), "Calling slice_features with one dataset is only"
            " valid for mode == 'total'."
            return X1[:, self.total_feature_slices[idx_1]].copy()
        s1, s2 = self.get_common_features(idx_1, idx_2)
        return X1[:, s1].copy(), X2[:, s2].copy()

    def assert_common_features(self, datasets: List[AnnData]) -> None:
        """
        For testing purposes, asserts a list of AnnData objects is correctly
        sliced in a common features space.
        """
        if self.mode == "total":
            com_features: Optional[np.ndarray] = None
            for adata, fslice in zip(datasets, self.total_feature_slices):
                if com_features is None:
                    com_features = adata.var_names[fslice].to_numpy()
                else:
                    np.testing.assert_array_equal(
                        com_features,
                        adata.var_names[fslice].to_numpy(),
                    )
        elif self.mode == "pairwise":
            for i, adata_i in enumerate(datasets):
                for j, adata_j in enumerate(datasets):
                    slice_i, slice_j = self.pairwise_feature_slices[i, j]
                    np.testing.assert_array_equal(
                        adata_i.var_names[slice_i].to_numpy(),
                        adata_j.var_names[slice_j].to_numpy(),
                    )
        else:
            raise ValueError(f"Unrecognized mode: {self.mode}.")
