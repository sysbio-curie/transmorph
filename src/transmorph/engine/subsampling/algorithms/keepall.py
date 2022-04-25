#!/usr/bin/env python3

import numpy as np

from typing import List, Tuple

from ..subsampling import Subsampling


class KeepAll(Subsampling):
    """
    Default subsampling scheme where all points are selected, equivalent
    to doing nothing.
    """

    def __init__(self):
        Subsampling.__init__(self, str_identifier="KEEPALL")

    def subsample(
        self, datasets: List[np.ndarray]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Selects all points.
        """
        return [
            (np.ones(X.shape[0]).astype(bool), np.arange(X.shape[0])) for X in datasets
        ]
