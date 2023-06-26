#!/usr/bin/env python3

from typing import List

import anndata as ad
import numpy as np
import scanpy.external as sce

from ..transformation import Transformation
from ....utils.matrix import extract_chunks


class Harmony(Transformation):
    """
    Wraps scanpy.external.harmony to be used within a transmorph pipeline as
    a Transformation.

    Parameters
    ----------
    **kwargs: Parameters to be passed to scanpy.external.harmony.
    See at
        https://scanpy.readthedocs.io/en/stable/generated/
        scanpy.external.pp.harmony_integrate.html

    Reference
    ---------

    [Korsunky 2019] https://www.nature.com/articles/s41592-019-0619-0
    """

    def __init__(self, **kwargs):
        Transformation.__init__(
            self,
            str_identifier="HARMONY",
            preserves_space=False,
        )
        self.kwargs = kwargs

    def check_input(self, datasets: List[np.ndarray]) -> None:
        assert len(datasets) > 0, "No datasets provided."
        if any(X.shape[1] > 100 for X in datasets):
            self.warn("Make sure you reduced dimension before Harmony.")

    def transform(
        self,
        datasets: List[ad.AnnData],
        embeddings: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Runs HarmonyPy and returns the embedding result.
        """
        adata_all = ad.concat(datasets, label="batch")
        adata_all.obsm["harmony_tmp"] = np.concatenate(embeddings, axis=0)
        # Will raise an exception if harmonypy is not installed
        sce.pp.harmony_integrate(
            adata_all,
            key="batch",
            basis="harmony_tmp",
            **self.kwargs,
        )
        return extract_chunks(
            adata_all.obsm["X_pca_harmony"],
            [adata.n_obs for adata in datasets],
        )
