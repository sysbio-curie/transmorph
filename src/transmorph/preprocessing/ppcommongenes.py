#!/usr/bin/env python3

from typing import List, Optional
from anndata import AnnData

import numpy as np

from .preprocessingABC import PreprocessingABC
from ..utils.anndata_interface import common_genes, get_matrix


class PPCommonGenes(PreprocessingABC):
    """
    Puts anndata objects in their larger common gene space. Only acts
    on AnnData.X matrix, so this must be done very early in a pipeline.
    """

    def __init__(self, n_top_var: Optional[int] = None):
        super().__init__()
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
            print(f"PPCG > {self.n_genes} genes kept.")
        for adata in datasets:
            if X_kw == "":
                results.append(adata[:, cgenes].X)
            else:
                X = get_matrix(adata, X_kw)
                assert X.shape[1] == adata.n_vars, (
                    f"Inconsistent number of features in {X_kw} representation."
                    f"Expected {adata.n_vars}, found {X.shape[1]}."
                )
                all_pos = np.zeros(cgenes.shape, dtype=int)
                for i, gene in enumerate(cgenes):
                    all_pos[i] = adata.var_names.get_loc(gene)
                results.append(X[:, all_pos])
        return results
