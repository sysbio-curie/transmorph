#!/usr/bin/env python3

from typing import List
from anndata import AnnData

from ..utils.anndata_interface import isset_matrix


class CheckingABC:
    """
    TODO
    """

    def __init__(
        self,
        threshold: float = 0.0,
        accept_if_lower: bool = True,
        verbose: bool = False,
    ):
        self.threshold = threshold
        self.accept_if_lower = accept_if_lower
        self.verbose = verbose
        self.last_value = 0.0

    def verify_input(self, adata: AnnData, X_kw: str = "") -> None:
        """
        Raises an exception if there is missing information.
        """
        assert isset_matrix(adata, X_kw), f"No key {X_kw} found."

    def check(self, datasets: List[AnnData], X_kw: str = "") -> bool:
        for adata in datasets:
            self.verify_input(adata, X_kw)
        self.last_value = self.evaluate_metric(datasets, X_kw)
        if self.accept_if_lower:
            return self.last_value <= self.threshold
        return self.last_value >= self.threshold

    def evaluate_metric(self, datasets: List[AnnData], X_kw: str = "") -> float:
        raise NotImplementedError
