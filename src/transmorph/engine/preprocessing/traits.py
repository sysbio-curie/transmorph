#!/usr/bin/env python3

import numpy as np

from anndata import AnnData
from typing import List

from transmorph.engine.preprocessing import Preprocessing
from transmorph.engine.traits import (
    HasMetadata,
    IsRepresentable,
    UsesCommonFeatures,
    assert_trait,
)
from transmorph.utils.type import assert_type


class IsPreprocessable:
    """
    A preprocessable object is a layer that can contain internal
    preprocessing steps.
    """

    def __init__(self) -> None:
        self.preprocessings: List[Preprocessing] = []

    @property
    def has_preprocessings(self) -> bool:
        return len(self.preprocessings) > 0

    @property
    def preserves_space(self) -> bool:
        return all(pp.preserves_space for pp in self.preprocessings)

    def add_preprocessing(self, preprocessing: Preprocessing) -> None:
        """
        Adds a preprocessing step to the layer, that will be applied
        before running the internal algorithm.
        """
        assert_type(preprocessing, Preprocessing)
        assert preprocessing not in self.preprocessings, "Preprocessing already exists."
        self.preprocessings.append(preprocessing)

    def preprocess(
        self, datasets: List[AnnData], representer: IsRepresentable
    ) -> List[np.ndarray]:
        """
        Runs all preprocessings.
        """
        is_feature_space = representer.is_feature_space
        assert_trait(representer, IsRepresentable)
        Xs = [representer.get(adata) for adata in datasets]
        for preprocessing in self.preprocessings:
            # If necessary, we let preprocessing retrieve
            # additional information
            if isinstance(preprocessing, HasMetadata):
                preprocessing.retrieve_all_metadata(datasets)
            if isinstance(preprocessing, UsesCommonFeatures):
                preprocessing.retrieve_common_features(datasets, is_feature_space)
            Xs = preprocessing.transform(Xs)
            is_feature_space = preprocessing.preserves_space
        return Xs
