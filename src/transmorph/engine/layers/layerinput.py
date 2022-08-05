#!/usr/bin/env python3

from anndata import AnnData
from typing import List

from . import Layer
from ..traits import IsRepresentable
from ...utils.anndata_manager import anndata_manager as adm, AnnDataKeyIdentifiers


class LayerInput(Layer, IsRepresentable):
    """
    Every integration pipeline must contain exactly one input layer,
    followed by an arbitrary network structure. Every
    pipeline is initialized using this input layer. This
    layer is the first to be called by any model. Its role
    is just to check all representations are present, and
    call subsequent layers. This layer can provide AnnData
    matrix representations.
    """

    def __init__(self) -> None:
        Layer.__init__(self, compatible_inputs=[], str_identifier="INPUT")
        IsRepresentable.__init__(
            self, repr_key=AnnDataKeyIdentifiers.BaseRepresentation
        )

    def fit(self, datasets: List[AnnData]) -> List[Layer]:
        """
        Simply checks dataset representations are at the right
        place, and call downstream layers.

        Parameters
        ----------
        datasets: List[AnnData]
            Datasets to run checking on.
        """
        self.log("Checking if all representations are present.")
        # Detecting if datasets are in feature space
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
        """
        Each layer input is its own embedding reference, for
        obvious structural reasons.
        """
        return self
