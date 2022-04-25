#!/usr/bin/env python3

from anndata import AnnData
from typing import List

from . import Layer
from ..traits import CanCatchChecking, IsRepresentable
from ...utils import AnnDataKeyIdentifiers


class LayerOutput(Layer, CanCatchChecking, IsRepresentable):
    """
    Simple layer to manage network outputs. There cannot be several output layers.
    for now, but it is a TODO
    """

    def __init__(self) -> None:
        Layer.__init__(
            self,
            compatible_inputs=[IsRepresentable],
            str_identifier="OUTPUT",
        )
        IsRepresentable.__init__(
            self,
            repr_key=AnnDataKeyIdentifiers.TransmorphRepresentation,
        )
        CanCatchChecking.__init__(self)

    def fit(self, datasets: List[AnnData]) -> List[Layer]:
        """
        Simply retrieves latest data representation, and stores it
        under obsm["transmorph"] key.
        """
        for adata in datasets:
            X = self.embedding_reference.get_representation(adata)
            self.write_representation(
                adata,
                X,
                self.embedding_reference.is_feature_space,
                persist="output",
            )
        return []
