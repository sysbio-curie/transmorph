#!/usr/bin/env python3

import numpy as np

import anndata as ad
from typing import List

from ..transformation import Transformation
from ...traits.usescommonfeatures import UsesCommonFeatures
from ....utils.dimred import ica
from ....utils.matrix import extract_chunks


class ICA(Transformation, UsesCommonFeatures):
    """
    Embeds a set of datasets in a common independent component (IC) space,
    using StabilizedICA package. This is typically slower than PCA, but
    features better tend to represent relevant and interpretable signals.

    Parameters
    ----------
    n_components: int, default = 30
        Number of PCs to use.

    max_iter: int, default = 1000
        Number of iterations in the optimization procedure.

    n_runs: int, default = 10
        Number of runs to carry out, more runs will yield a more
        stable result.
    """

    def __init__(
        self,
        n_components: int = 30,
        max_iter: int = 1000,
        n_runs: int = 10,
    ):
        Transformation.__init__(self, str_identifier="PCA", preserves_space=False)
        UsesCommonFeatures.__init__(self, mode="total")
        self.n_components = n_components
        self.max_iter = max_iter
        self.n_runs = n_runs

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
        X_ica = ica(
            np.concatenate(to_reduce, axis=0),
            n_components=self.n_components,
            max_iter=self.max_iter,
            n_runs=self.n_runs,
        )
        return extract_chunks(X_ica, [X.shape[0] for X in datasets])
