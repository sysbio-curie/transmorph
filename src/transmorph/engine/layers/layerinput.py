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
        self.log("Checking all representations are present.")
        for adata in datasets:
            assert (
                adm.get_value(adata, self.repr_key) is not None
            ), f"Representation {self.repr_key} missing in {adata}."
        self.log("All representations found. Continuing.")
        return self.output_layers
