#!/usr/bin/env python3

from __future__ import annotations

import logging

from anndata import AnnData
from typing import List, Optional

from . import Layer
from ..matching import Matching, _TypeMatchingSet
from ..subsampling import (
    Subsampling,
    KeepAll,
)
from ..traits import (
    ContainsTransformations,
    HasMetadata,
    IsProfilable,
    IsSubsamplable,
    IsRepresentable,
    IsWatchable,
    UsesCommonFeatures,
    UsesReference,
)
from ..watchers import WatcherMatching, WatcherTiming
from ... import profile_method


class LayerMatching(
    Layer,
    ContainsTransformations,
    IsWatchable,
    IsProfilable,
    IsSubsamplable,
):
    """
    This layer performs a matching between two or more datasets.
    It wraps an object derived from MatchingABC.
    """

    def __init__(self, matching: Matching, subsampling: Optional[Subsampling]) -> None:
        Layer.__init__(
            self,
            compatible_inputs=[IsRepresentable],
            str_identifier="MATCHING",
        )
        if subsampling is None:
            subsampling = KeepAll()
        IsSubsamplable.__init__(self, subsampling)
        IsWatchable.__init__(self, compatible_watchers=[WatcherMatching, WatcherTiming])
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
        if self.has_transformations:
            self.info("Calling preprocessings.")
        Xs = self.transform(datasets, self.embedding_reference)
        if not isinstance(self.subsampling, KeepAll):
            self.info("Calling subsampling.")
        # Subsampling
        self.subsample(datasets=datasets, matrices=Xs)
        Xs = self.slice_matrices(datasets=datasets, matrices=Xs)
        # Matching
        self.info("Calling matching.")
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
            self.matching.get_reference_index(datasets)
        self.matching.check_input(Xs)
        self.matching_matrices = self.matching.fit(Xs)
        # Trimming? Extrapolating?
        self.log("Fitted.", level=logging.INFO)
        return self.output_layers

    def get_matchings(self) -> _TypeMatchingSet:
        """
        Returns computed matchings for read-only purposes.
        """
        assert self.matching_matrices is not None, "Layer is not fit."
        return self.matching_matrices
