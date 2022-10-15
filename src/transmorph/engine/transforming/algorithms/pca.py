#!/usr/bin/env python3

from typing import Literal, List

import anndata as ad
import numpy as np

from ..transformation import Transformation
from ...traits.usescommonfeatures import UsesCommonFeatures
from ...traits.usesreference import UsesReference
from ....utils.dimred import pca, pca_projector


class PCA(Transformation, UsesCommonFeatures, UsesReference):
    """
    Embeds a set of datasets in a common PC space, following one of the following
    strategies:

        - "concatenate": concatenate all datasets together on the axis 0, then
        perform a PCA on this result. Needs all datasets to be in the same
        features space.

        - "reference": project everything on the first dataset PC space. Needs
        all datasets to be in the same features space.

        - "independent": assume variance axes are preserved between datasets, and
        perform an independent PC projection of same dimensionality for each dataset.

    Parameters
    ----------
    n_components: int, default = 30
        Number of PCs to use.

    strategy: str, default = 'concatenate'
        Strategy to choose projection space in 'concatenate', 'reference',
        'composite' and 'independent'
    """

    def __init__(
        self,
        n_components: int = 30,
        strategy: Literal["concatenate", "reference", "independent"] = "concatenate",
    ):
        Transformation.__init__(self, str_identifier="PCA", preserves_space=False)
        UsesCommonFeatures.__init__(self, mode="total")
        UsesReference.__init__(self)
        self.n_components = n_components
        self.strategy = strategy

    def check_input(self, datasets: List[np.ndarray]) -> None:
        assert len(datasets) > 0, "No datasets provided."
        if self.strategy != "independent":
            assert all(
                X.shape[1] == datasets[0].shape[1] for X in datasets
            ), f"All datasets must be of same dimension for strategy={self.strategy}."

    def transform(
        self,
        datasets: List[ad.AnnData],
        embeddings: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Slices datasets in the same space if necessary, then carries out the
        information.
        """
        to_reduce = []
        for i, X in enumerate(embeddings):
            to_reduce.append(self.slice_features(X1=X, idx_1=i))
        if to_reduce[0].shape[1] <= self.n_components:
            return to_reduce

        # TODO: check if X_pca is set
        if self.strategy == "independent":
            return [pca(X, self.n_components) for X in to_reduce]

        pca_obj = None
        if self.strategy == "concatenate":
            pca_obj = pca_projector(
                np.concatenate(to_reduce, axis=0), self.n_components
            )
        elif self.strategy == "reference":
            pca_obj = pca_projector(
                self.get_reference_item(to_reduce), self.n_components
            )
        else:
            raise NotImplementedError(f"Unknown strategy for PCA {self.strategy}.")

        return [pca_obj.transform(X) for X in to_reduce]
