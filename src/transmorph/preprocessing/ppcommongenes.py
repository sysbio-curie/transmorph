#!/usr/bin/env python3

from typing import List
from anndata import AnnData

import numpy as np

from .preprocessingABC import PreprocessingABC


class PPCommonGenes(PreprocessingABC):
    """
    Puts anndata objects in their larger common gene space. Only acts
    on AnnData.X matrix, so this must be done very early in a pipeline.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def transform(self, datasets: List[AnnData], X_kw: str = "") -> List[np.ndarray]:
        results = []
        common_genes = datasets[0].var_names
        for adata in datasets[1:]:
            common_genes = common_genes.intersection(adata.var_names)
        if self.verbose:
            print(f"{len(common_genes)} genes kept.")
        for adata in datasets:
            results.append(adata[:, common_genes].X)
        return results
