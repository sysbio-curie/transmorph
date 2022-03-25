#!/usr/bin/env python3

from abc import abstractmethod
from anndata import AnnData
from typing import List, Dict

from transmorph.utils.anndata_interface import get_matrix, isset_matrix, set_matrix

import numpy as np


class SubsamplingABC:
    """
    A subsampling scheme choses a subset of representers from a large collection
    of points. These representers are then used during heavy computational parts
    of the pipeline to speed up computations. SubsamplingABC is the abstraction
    that allows other parts of Transmorph to manipulate subsampling schemes.
    """

    def __init__(self):
        pass

    @abstractmethod
    def _subsample_one(self, adata: AnnData, X_kw: str = "") -> Dict[str, np.ndarray]:
        """
        Every subsampling must override this method.

        Parameters
        ----------
        adata: AnnData
            Dataset to subsample from.

        X_kw: str, default = ""
            Identifier of the matrix representation to use. Use "" to use adata.X.

        Returns
        -------
        A dictionary of shape {"is_anchor": $A, "ref_anchor": $B} where
        - $A is a boolean vector of same size as the dataset, where $A[i] is true
          if adata_i is a representer.
        - $B is an integer vector of the same size as the dataset, where $B[i]
          contains the index of the representer of i (can be itself).
        """
        raise NotImplementedError

    def subsample(self, datasets: List[AnnData], X_kw: str = "") -> None:
        """
        Applies _subsample_one to all datasets if necessary
        """
        for adata in datasets:  # Just flag everything as an anchor
            if self.is_computed(adata):
                continue
            result = self._subsample_one(adata, X_kw)
            self.set_anchors(adata, result["is_anchor"])
            self.set_references(adata, result["ref_anchor"])

    def is_computed(self, adata: AnnData) -> bool:
        """
        Returns true if adata has been subsampled.
        """
        return isset_matrix(adata, "is_anchor")

    def get_anchors(self, adata: AnnData) -> np.ndarray:
        """
        Returns the list of anchors as a boolean vector.
        """
        assert isset_matrix(adata, "is_anchor"), "Subsampling not computed."
        return get_matrix(adata, "is_anchor")

    def get_references(self, adata: AnnData) -> np.ndarray:
        """
        Returns the list of referent anchors as an integer vector.
        """
        assert isset_matrix(adata, "ref_anchor"), "Subsampling not computed."
        return get_matrix(adata, "ref_anchor")

    def set_anchors(self, adata: AnnData, is_anchor: np.ndarray):
        """
        To use in implementations to set the anchors boolean vector.
        """
        set_matrix(adata, "is_anchor", is_anchor)

    def set_references(self, adata: AnnData, ref_anchor: np.ndarray):
        """
        To use in implementations to set the reference integer vector.
        """
        set_matrix(adata, "ref_anchor", ref_anchor)
