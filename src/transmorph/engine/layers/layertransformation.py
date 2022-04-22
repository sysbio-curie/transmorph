#!/usr/bin/env python3

import logging

from anndata import AnnData
from typing import List

from . import Layer
from ..traits import (
    ContainsTransformations,
    IsProfilable,
    profile_method,
    IsRepresentable,
)


class LayerTransformation(
    Layer, ContainsTransformations, IsProfilable, IsRepresentable
):
    """
    This layer encapsulates a series of preprocessing algorithms derived
    from PreprocessingABC.
    """

    def __init__(self) -> None:
        Layer.__init__(
            self,
            compatible_inputs=[IsRepresentable],
            str_identifier="PREPROCESSING",
        )
        IsRepresentable.__init__(self, repr_key=f"{self}_{self.layer_id}")

    @profile_method
    def fit(self, datasets: List[AnnData]) -> List[Layer]:
        """
        Simply runs preprocessing algorithms and returns the result.
        """
        if self.has_transformations:
            self.log("Calling transformations.", level=logging.INFO)
        Xs = self.transform(datasets, self.embedding_reference)
        is_feature_space = (
            self.embedding_reference.is_feature_space and self.preserves_space
        )
        for adata, X_after in zip(datasets, Xs):
            self.write_representation(adata, X_after, is_feature_space=is_feature_space)
        self.log("Done.", level=logging.INFO)
        return self.output_layers
