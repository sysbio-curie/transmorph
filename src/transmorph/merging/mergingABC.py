#!/usr/bin/env python3

import warnings
import numpy as np

from abc import ABC, abstractmethod
from anndata import AnnData
from scipy.sparse import csr_matrix
from typing import List, Optional, Union

from transmorph.utils.anndata_interface import isset_matrix

from ..matching.matchingABC import MatchingABC


class MergingABC(ABC):
    """
    A Merging is a class containing a function merge(x1, ..., xn, m1, ..., mk)
    with n >= 2 and k >= 1, where x1, ..., xn are n datasets and m(i(i-1)/2+j)
    contains the matching between xi and xj. merge() outputs a common representation
    for all samples from datasets x1 to xn that can be an embedding or a network.

    Parameters
    ----------
    matching: MatchingABC
        Fitted, referenced matching between datasets.
    """

    def __init__(self, use_reference: bool = False):
        self.use_reference = use_reference
        self.input_checked = False

    def _check_input(
        self,
        datasets: List[AnnData],
        matching: Optional[MatchingABC] = None,
        matching_mtx: Optional[Union[csr_matrix, np.ndarray]] = None,
        X_kw: str = "",
        reference_idx: int = -1,
    ) -> None:
        """
        Checking if number of matchings and datasets coincides with reference strategy.
        This method is automatically called at the beginning MergingABC._check_input().
        Any class inheriting from MergingABC can add rules to this method.
        """
        if matching is None and matching_mtx is None:
            raise ValueError("No matching provided.")
        if self.use_reference:
            assert reference_idx >= 0, "Missing reference dataset."
        else:
            if reference_idx >= 0:
                warnings.warn("Ignoring reference provided to a ref-less merging.")
        if matching_mtx is not None:
            assert (
                len(datasets) == 2
            ), "Directly passing a matching only available with two datasets."
            assert (
                matching is None
            ), "Ambiguous use of both a Matching and a matching matrix."
        for adata in datasets:
            assert isset_matrix(adata, X_kw), f"KeyError: {X_kw}"
        self.input_checked = True

    @abstractmethod
    def fit(
        self,
        datasets: List[AnnData],
        matching: Optional[MatchingABC] = None,
        matching_mtx: Optional[Union[csr_matrix, np.ndarray]] = None,
        X_kw: str = "",
        reference_idx: int = -1,
    ) -> List[np.ndarray]:
        pass
