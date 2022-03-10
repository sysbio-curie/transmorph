#!/usr/bin/env python3

from typing import List
from anndata import AnnData


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

    def check(self, datasets: List[AnnData], representation_kw: str) -> bool:
        if self.accept_if_lower:
            return self.evaluate_metric(datasets) <= self.threshold
        return self.evaluate_metric(datasets) >= self.threshold

    def evaluate_metric(self, datasets: List[AnnData]) -> float:
        raise NotImplementedError
