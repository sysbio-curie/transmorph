#!/usr/bin/env python3

from __future__ import annotations

from anndata import AnnData
from typing import List, Optional

from . import Layer
from ..matching import Matching, _TypeMatchingSet
from ..subsampling import Subsampling
from ..traits import (
    CanCatchChecking,
    ContainsTransformations,
    HasMetadata,
    IsProfilable,
    profile_method,
    IsSubsamplable,
    IsRepresentable,
    UsesCommonFeatures,
    UsesReference,
    UsesSampleLabels,
)


class LayerMatching(
    Layer,
    CanCatchChecking,
    ContainsTransformations,
    IsProfilable,
    IsSubsamplable,
):
    """
    This layer performs a matching between two or more datasets.
    It wraps an object derived from MatchingABC.
    """

    def __init__(
        self, matching: Matching, subsampling: Optional[Subsampling] = None
    ) -> None:
        Layer.__init__(
            self,
            compatible_inputs=[IsRepresentable],
            str_identifier="MATCHING",
        )
        CanCatchChecking.__init__(self)
        ContainsTransformations.__init__(self)
        IsProfilable.__init__(self)
        IsSubsamplable.__init__(self, subsampling)
        self.matching = matching
        self.matching_matrices: Optional[_TypeMatchingSet] = None

    @profile_method
    def fit(self, datasets: List[AnnData]) -> List[Layer]:
        """
        Calling self.matching.fit after carrying out the requested
        preprocessings.
        """
        self.datasets = datasets.copy()  # Keeping a copy to preserve order
        # Preprocessing
        Xs = self.transform(
            datasets=datasets,
            representer=self.embedding_reference,
            log_callback=self.log,
        )
        # Subsampling
        if self.has_subsampling:
            self.info("Subsampling datasets...")
        self.subsample(datasets=datasets, matrices=Xs, log_callback=self.log)
        Xs = self.slice_matrices(datasets=datasets, matrices=Xs)
        # Matching
        self.info(f"Calling matching {self.matching}.")
        if isinstance(self.matching, HasMetadata):  # Metadata gathering
            self.matching.retrieve_all_metadata(datasets)
        if isinstance(self.matching, UsesCommonFeatures):  # Common features slicing
            is_feature_space = (
                self.embedding_reference.is_feature_space and self.preserves_space
            )
            self.matching.retrieve_common_features(
                datasets,
                is_feature_space=is_feature_space,
            )
        # Checks if there is a reference dataset
        if isinstance(self.matching, UsesReference):
            self.matching.retrieve_reference_index(datasets)
        if isinstance(self.matching, UsesSampleLabels):
            self.matching.retrieve_labels(datasets)
            self.matching.apply_subsampling_to_labels(self, datasets)
        self.matching.check_input(Xs)

        # Supersampling matrices
        self.matching_matrices = {}
        for key, T in self.matching.fit(Xs).items():
            i, j = key
            T = self.supersample_matrix(datasets[i], T, datasets[j])
            self.matching_matrices[i, j] = T

        # Trimming? Extrapolating?
        return self.output_layers

    def get_matchings(self) -> _TypeMatchingSet:
        """
        Returns computed matchings for read-only purposes.
        """
        assert self.matching_matrices is not None, "Layer is not fit."
        return self.matching_matrices
