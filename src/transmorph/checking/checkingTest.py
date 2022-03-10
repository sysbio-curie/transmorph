#!/usr/bin/env python3

# ONLY FOR TESTING PURPOSES

from anndata import AnnData
from typing import List

from .checkingABC import CheckingABC


class CheckingTest(CheckingABC):
    def __init__(self):
        super().__init__(threshold=0.0)

    def evaluate_metric(self, datasets: List[AnnData]) -> float:
        return 1.0  # Always fails
