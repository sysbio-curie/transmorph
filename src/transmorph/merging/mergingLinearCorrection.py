#!/usr/bin/env python3
#
from scipy.sparse import csr_matrix

from ..matching.matchingABC import MatchingABC
from mergingABC import MergingABC

import numpy as np
from typing import Union, List


class MergingLinearCorrection(MergingABC):
    """ """

    def __init__(self):
        pass

    def merge(
        self, datasets: List[np.ndarray], matching: MatchingABC
    ) -> Union[np.ndarray, csr_matrix]:
        pass
