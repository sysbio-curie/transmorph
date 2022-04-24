#!/usr/bin/env python3

from anndata import AnnData
from typing import List

from . import Layer
from ..traits import IsRepresentable
from ...utils import anndata_manager as adm, AnnDataKeyIdentifiers


class LayerInput(Layer, IsRepresentable):
    """
    Every pipeline must contain exactly one input layer, followed by an
    arbitrary network structure. Every pipeline is initialized using this
    input layer.
    """

    def __init__(self) -> None:
        Layer.__init__(self, compatible_inputs=[], str_identifier="INPUT")
        IsRepresentable.__init__(
            self, repr_key=AnnDataKeyIdentifiers.BaseRepresentation
        )

    def fit(self, datasets: List[AnnData]) -> List[Layer]:
        """
        Simply calls the downstream layers.
        """
        self.log("Checking if all representations are present.")
        self.is_feature_space = True
        for adata in datasets:
            X = adm.get_value(adata, self.repr_key)
            assert X is not None, f"Representation {self.repr_key} missing in {adata}."
            if X is not adata.X:
                self.is_feature_space = False
        self.log(
            f"All representations found, in feature space: {self.is_feature_space}."
        )
        return self.output_layers

    @property
    def embedding_reference(self) -> IsRepresentable:
        return self
