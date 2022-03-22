#!/usr/bin/env python3

from typing import List
from anndata import AnnData

import numpy as np

from .preprocessingABC import PreprocessingABC
from ..utils.anndata_interface import get_matrix
from ..utils.dimred import pca_multi


class PPPCA(PreprocessingABC):
    """
    TODO
    """

    def __init__(self, n_components: int = 2, strategy: str = "concatenate"):
        self.n_components = n_components
        self.strategy = strategy

    def transform(self, datasets: List[AnnData], X_kw: str = "") -> List[np.ndarray]:
        to_reduce = []
        for adata in datasets:
            to_reduce.append(get_matrix(adata, X_kw))
        return pca_multi(
            to_reduce,
            n_components=self.n_components,
            strategy=self.strategy,
            return_transformation=False,
        )
