#!/usr/bin/env python3

from anndata import AnnData
from typing import List

from transmorph.engine.traits.cancatchchecking import CanCatchChecking

from . import Layer
from ..traits import (
    ContainsTransformations,
    IsProfilable,
    profile_method,
    IsRepresentable,
)


class LayerTransformation(
    Layer,
    CanCatchChecking,
    ContainsTransformations,
    IsProfilable,
    IsRepresentable,
):
    """
    A LayerTransformation encapsulates a Transformation algorithm,
    which is useful to alter representations of datasets,  facilitating
    the work of subsequent algorithms. It contains a set of
    transformations which will be applied in the order of addition.
    This layer can provide a matrix representation of AnnData
    objects.
    """

    def __init__(self) -> None:
        Layer.__init__(
            self,
            compatible_inputs=[IsRepresentable],
            str_identifier="TRANSFORMATION",
        )
        IsRepresentable.__init__(self, repr_key=f"{self}_{self.layer_id}")
        ContainsTransformations.__init__(self)
        IsProfilable.__init__(self)
        CanCatchChecking.__init__(self)

    @profile_method
    def fit(self, datasets: List[AnnData]) -> List[Layer]:
        """
        Sequentially runs the internal algorithms. Then, returns
        next layers in the model.

        Parameters
        ----------
        datasets: List[AnnData]
            Datasets to run merging on.
        """
        self.log(f"Retrieving data from {self.embedding_reference.repr_key}.")
        Xs = self.transform(
            datasets=datasets,
            representer=self.embedding_reference,
            log_callback=self.log,
        )
        is_feature_space = (
            self.embedding_reference.is_feature_space and self.preserves_space
        )
        for adata, X_after in zip(datasets, Xs):
            self.write_representation(adata, X_after, is_feature_space=is_feature_space)
        return self.output_layers
