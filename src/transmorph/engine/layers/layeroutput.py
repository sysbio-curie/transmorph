#!/usr/bin/env python3

from anndata import AnnData
from typing import List

from . import Layer
from ..traits import CanCatchChecking, IsRepresentable
from ...utils.anndata_manager import AnnDataKeyIdentifiers


class LayerOutput(Layer, CanCatchChecking, IsRepresentable):
    """
    A LayerOutput is the final step of any model. Its only role
    is to retrieve last computed representation, and write it
    durably in AnnDatas under the entry .obsm['X_transmorph'].
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
        self.log(f"Retrieving data from {self.embedding_reference.repr_key}.")
        for adata in datasets:
            self.write_representation(
                adata,
                self.embedding_reference.get_representation(adata),
                self.embedding_reference.is_feature_space,
                persist="output",
            )
        return []
