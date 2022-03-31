#!/usr/bin/env python3

from typing import List
from anndata import AnnData

from ..utils.anndata_interface import isset_matrix


class CheckingABC:
    """
    CheckingABC is the abstract class used to describe checking algorithms.
    A "checking" is a function that takes as input a set of datasets
    endowed with their respective embeddings, and provides a boolean
    answer to "are these datasets satisfyingly integrated?". This answer
    is computed using statistical testings and a threshold. All Checking
    algorithms must inherit CheckingABC.

    Parameters
    ----------
    threshold: float, default = 0.0
        Threshold value to compare statistical test result with. By
        default, checking passes if test value is below the threshold.

    accept_if_lower: bool, default = True
        If True, then lower is better. Otherwise, higher is better.

    Attributes
    ----------
    last_value: float
        Value returned by the last call to the testing algorithm. Can
        be retrived by other modules for logging purposes for instance.
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
        self.last_value: float = 0.0
        self.input_verified = False

    def verify_input(self, adata: AnnData, X_kw: str = "") -> None:
        """
        Checks input validity, can be overriden at the condition to
        start by calling the base verify_input inside.
        """
        assert isset_matrix(adata, X_kw), f"No key {X_kw} found."
        self.input_verified = True  # Flags the base method has been called

    def check(self, datasets: List[AnnData], X_kw: str = "") -> bool:
        """
        Performs the checking over a list of AnnData, returns if the
        statistical test is considered accepted given specified threshold.
        This method should not be overriden, see $evaluate_metric instead.

        Parameters
        ----------
        datasets: List[AnnData]
            List of target datasets.

        X_kw: str, default = ""
            Target embeddings location in AnnDatas.
        """
        for adata in datasets:
            self.verify_input(adata, X_kw)
        assert self.input_verified, "Child class did not call CheckingABC.verify_input."
        self.last_value = self.evaluate_metric(datasets, X_kw)
        if self.accept_if_lower:
            return self.last_value <= self.threshold
        return self.last_value >= self.threshold

    def evaluate_metric(self, datasets: List[AnnData], X_kw: str = "") -> float:
        """
        Computes and return a test value from a set of datasets. Must
        be overriden by every child classes.

        Parameters
        ----------
        datasets: List[AnnData]
            List of target datasets.

        X_kw: str, default = ""
            Target embeddings location in AnnDatas.
        """
        raise NotImplementedError
