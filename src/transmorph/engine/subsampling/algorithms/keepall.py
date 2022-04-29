#!/usr/bin/env python3

import numpy as np

from typing import List

from ..subsampling import Subsampling, _TypeSubsampling


class KeepAll(Subsampling):
    """
    This is the identity subsampling, which does nothing and serves
    as a placeholder when no subsampling is needed.
    """

    def __init__(self):
        Subsampling.__init__(self, str_identifier="KEEPALL")

    def subsample(self, datasets: List[np.ndarray]) -> List[_TypeSubsampling]:
        """
        Selects all points, and defines each point as its own anchor.
        """
        return [
            (np.ones(X.shape[0]).astype(bool), np.arange(X.shape[0])) for X in datasets
        ]
