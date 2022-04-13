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

    def transform(self, datasets: List[AnnData], X_kw: str = "") -> List[np.ndarray]:
        cgenes = common_genes(datasets, self.n_top_var)
        assert cgenes.shape[0] > 0, "No common gene found."
        results = []
        if self.verbose:
            print(f"{len(cgenes)} genes kept.")
        for adata in datasets:
            results.append(adata[:, cgenes].X)
        return results
