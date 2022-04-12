#!/usr/bin/env python3

from typing import List
from anndata import AnnData

import numpy as np

from .preprocessingABC import PreprocessingABC
from ..utils.anndata_interface import common_genes


class PPCommonGenes(PreprocessingABC):
    """
    Puts anndata objects in their larger common gene space. Only acts
    on AnnData.X matrix, so this must be done very early in a pipeline.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def transform(self, datasets: List[AnnData], X_kw: str = "") -> List[np.ndarray]:
        cgenes = common_genes(datasets)
        results = []
        if self.verbose:
            print(f"{len(cgenes)} genes kept.")
        for adata in datasets:
            results.append(adata[:, cgenes].X)
        return results
