#!/usr/bin/env python3

from __future__ import annotations

import logging
import numpy as np

from anndata import AnnData
from scipy.sparse import csr_matrix
from typing import Dict, List, Literal, Optional, Tuple

from transmorph.engine import Layer
from transmorph.engine.matching import Matching, WatcherMatching
from transmorph.engine.preprocessing import IsPreprocessable
from transmorph.engine.profiler import IsProfilable, profile_method
from transmorph.engine.subsampling import (
    IsSubsamplable,
    SubsamplingKeepAll,
    SubsamplingABC,
)
from transmorph.engine.traits import (
    HasMetadata,
    IsRepresentable,
    UsesCommonFeatures,
    UsesReference,
)
from transmorph.engine.watchers import IsWatchable, WatcherTiming


class LayerMatching(Layer, IsPreprocessable, IsWatchable, IsProfilable, IsSubsamplable):
    """
    This layer performs a matching between two or more datasets.
    It wraps an object derived from MatchingABC.
    """

    def __init__(
        self, matching: Matching, subsampling: Optional[SubsamplingABC]
    ) -> None:
        Layer.__init__(
            self,
            compatible_inputs=[IsRepresentable],
            str_type="MATCHING",
        )
        IsPreprocessable.__init__(self)
        IsProfilable.__init__(self)
        if subsampling is None:
            subsampling = SubsamplingKeepAll()
        IsSubsamplable.__init__(self, subsampling)
        IsWatchable.__init__(self, compatible_watchers=[WatcherMatching, WatcherTiming])
        self.matching = matching
        self.matching_matrices: Dict[Tuple[int, int], csr_matrix] = {}
        self.datasets: List[AnnData] = []

    @property
    def n_datasets(self) -> int:
        return len(self.datasets)

    @profile_method
    def fit(self, datasets: List[AnnData]) -> List[Layer]:
        """
        Calling self.matching.fit after carrying out the requested
        preprocessings.
        """
        self.datasets = datasets.copy()  # Keeping a copy to preserve order
        if self.has_preprocessings:
            self._log("Calling preprocessings.", level=logging.INFO)
        Xs = self.preprocess(datasets, self.embedding_reference)
        if not isinstance(self.subsampling, SubsamplingKeepAll):
            self._log("Calling subsampling.", level=logging.INFO)
        self.subsample(datasets=datasets, matrices=Xs)
        Xs = self.slice_matrices(datasets=datasets, matrices=Xs)
        self._log("Calling matching.", level=logging.INFO)
        if isinstance(self.matching, HasMetadata):
            self.matching.retrieve_all_metadata(datasets)
        if isinstance(self.matching, UsesCommonFeatures):
            self.matching.retrieve_common_features(datasets)
        # Checks if there is a reference dataset
        ref_id = UsesReference.get_reference_index(datasets)
        self.matching.check_input(Xs)
        self.matching.fit(Xs, reference_idx=ref_id)
        # Trimming? Extrapolating?
        self._log("Fitted.", level=logging.INFO)
        return self.output_layers

    def get_adata_index(self, target: AnnData) -> int:
        """
        Returns the index of AnnData object passed as parameter.
        """
        for i, adata in enumerate(self.datasets):
            if adata is target:
                return i
        raise ValueError(f"{target} is an unknown dataset.")

    def get_matching(
        self,
        adata_1: int,
        adata_2: int,
        mode: Literal["raw", "boolean", "row_normalized"] = "row_normalized",
    ) -> csr_matrix:
        """
        Return the matching between two datasets. Throws an error
        if matching is not fitted, or if the required matching does not exist.

        Parameters
        ----------
        ind_1: ind
            Index of the first dataset

        ind_2: ind
            Index of the second dataset

        mode: Literal["raw", "boolean", "row_normalized"],
              default = "row_normalized"
            Transformation to apply to the matching matrix
            - "raw": Return the matching without modifications
            - "boolean": Return a boolean version of the matching,
              where all nonzero entries are set to True
            - "raw_normalized": Return a row normalized version of
              the matching, where all nonzero rows marginalize to 1.

        Returns
        -------
        T = (n_ind1, n_ind2) CSR sparse matrix, where Tkl is the
        matching strength between X_ind1_k and X_ind2_l.
        """
        ind_1 = self.get_adata_index(adata_1)
        ind_2 = self.get_adata_index(adata_2)
        matching = self.matching_matrices.get((ind_1, ind_2), None)
        if matching is None:
            matching = self.matching_matrices.get((ind_2, ind_1), None)
            if matching is None:
                raise ValueError(
                    f"No matching found between indices {ind_1} and {ind_2}."
                )
            matching = csr_matrix(matching.T)
        if mode == "row_normalize":
            coefs = np.array(matching.sum(axis=1))
            coefs[coefs == 0.0] = 1.0
            return csr_matrix(matching / coefs)
        elif mode == "boolean":
            return matching.astype(bool)
        elif mode == "raw":
            return matching
        else:
            raise ValueError(f"Unrecognized mode {mode}.")
