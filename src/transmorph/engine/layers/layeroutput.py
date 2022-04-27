#!/usr/bin/env python3

from anndata import AnnData
from typing import List

from . import Layer
from ..traits import CanCatchChecking, IsRepresentable
from ...utils import AnnDataKeyIdentifiers


class LayerOutput(Layer, CanCatchChecking, IsRepresentable):
    """
    A LayerOutput is the final step of any model. Its only role
    is to retrieve last computed representation, and write it
    durably in AnnDatas under the entry .obsm['transmorph'].
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
        Simply retrieves last computed representation,
        and write it in AnnData objects.

        Parameters
        ----------
        datasets: List[AnnData]
            Datasets to write results in.
        """
        for adata in datasets:
            self.write_representation(
                adata,
                self.embedding_reference.get_representation(adata),
                self.embedding_reference.is_feature_space,
                persist="output",
            )
        return []
