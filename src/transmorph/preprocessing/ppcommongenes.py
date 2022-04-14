#!/usr/bin/env python3

from typing import List, Optional
from anndata import AnnData

import numpy as np

from .preprocessingABC import PreprocessingABC
from ..utils.anndata_interface import common_genes


class PPCommonGenes(PreprocessingABC):
    """
    Puts anndata objects in their larger common gene space. Only acts
    on AnnData.X matrix, so this must be done very early in a pipeline.
    """

    def __init__(self, n_top_var: Optional[int] = None, verbose: bool = False):
        self.verbose = verbose
        self.n_top_var = n_top_var
        self.n_genes = -1

    def transform(self, datasets: List[AnnData], X_kw: str = "") -> List[np.ndarray]:
        cgenes = common_genes(datasets, self.n_top_var)
        self.n_genes = cgenes.shape[0]
        assert self.n_genes > 0, (
            "No common gene found between datasets. "
            "Try increasing the number of available genes."
        )
        results = []
        if self.verbose:
            print(f"{self.n_genes} genes kept.")
        for adata in datasets:
            results.append(adata[:, cgenes].X)
        return results
