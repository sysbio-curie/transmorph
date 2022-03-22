#!/usr/bin/env python3

from abc import abstractmethod
from typing import List
from anndata import AnnData

import numpy as np


class PreprocessingABC:
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, datasets: List[AnnData], X_kw: str = "") -> List[np.ndarray]:
        pass
