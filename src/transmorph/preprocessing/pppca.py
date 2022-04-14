#!/usr/bin/env python3

from typing import Literal, List
from anndata import AnnData

import numpy as np

from .preprocessingABC import PreprocessingABC
from ..utils.anndata_interface import get_matrix
from ..utils.dimred import pca_multi


class PPPCA(PreprocessingABC):
    """
    Embeds a set of datasets in a common PC space, following one of the following
    strategies:

        - "concatenate": concatenate all datasets together on the axis 0, then
        perform a PCA on this result. Needs all datasets to be in the same
        features space.

        - "reference": project everything on the first dataset PC space. Needs
        all datasets to be in the same features space.

        - "composite": use an average of the transformation matrices to define
        projection PC space. Needs all datasets to be in the same features space.

        - "independent": assume variance axes are preserved between datasets, and
        perform an independent PC projection of same dimensionality for each dataset.

    Parameters
    ----------
    n_components: int, default = 2
        Number of PCs to use.

    strategy: str, default = 'concatenate'
        Strategy to choose projection space in 'concatenate', 'reference',
        'composite' and 'independent'
    """

    def __init__(
        self,
        n_components: int = 2,
        strategy: Literal["concatenate", "composite", "independent"] = "concatenate",
    ):
        super().__init__()
        self.n_components = n_components
        self.strategy = strategy

    def transform(self, datasets: List[AnnData], X_kw: str = "") -> List[np.ndarray]:
        to_reduce = []
        for adata in datasets:
            to_reduce.append(get_matrix(adata, X_kw))
        if self.verbose:
            print(f"PPPCA > Running PCA with {self.n_components} components...")
        return pca_multi(
            to_reduce,
            n_components=self.n_components,
            strategy=self.strategy,
            return_transformation=False,
        )
