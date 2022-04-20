#!/usr/bin/env python3

from typing import Literal, List

import numpy as np

from transmorph.engine.transforming import Transformation
from transmorph.engine.traits import UsesCommonFeatures
from transmorph.utils.dimred import pca_multi


class PCA(Transformation, UsesCommonFeatures):
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
        Transformation.__init__(self, str_identifier="PCA", preserves_space=False)
        UsesCommonFeatures.__init__(self, mode="total")
        self.n_components = n_components
        self.strategy = strategy

    def transform(self, datasets: List[np.ndarray]) -> List[np.ndarray]:
        """
        Slices datasets in the same space if necessary, then carries out the
        information.
        """
        to_reduce = []
        for i, X in enumerate(datasets):
            to_reduce.append(self.slice_features(X1=X, idx_1=i))
        return pca_multi(
            to_reduce,
            n_components=self.n_components,
            strategy=self.strategy,
            return_transformation=False,
        )
