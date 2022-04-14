#!/usr/bin/env python3

import numpy as np

from anndata import AnnData
from typing import Dict
from .subsamplingABC import SubsamplingABC


class SubsamplingKeepAll(SubsamplingABC):
    """
    Default subsampling scheme where all points are selected, equivalent
    to doing nothing.
    """

    def __init__(self):
        super().__init__()

    def _subsample_one(self, adata: AnnData, X_kw: str = "") -> Dict[str, np.ndarray]:
        n = adata.n_obs
        return {"is_anchor": np.ones(n).astype(bool), "ref_anchor": np.arange(n)}
