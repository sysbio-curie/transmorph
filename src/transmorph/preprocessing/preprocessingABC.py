#!/usr/bin/env python3

from abc import abstractmethod
from typing import List

import numpy as np


class PreprocessingABC:
    """
    Abstract class for Preprocessing objects. Implements a method
    transform(List[np.ndarray]) that computes the preprocessing.
    Child classes can be enriched by traits.
    """

    def __init__(self):
        pass

    @abstractmethod
    def transform(self, datasets: List[np.ndarray]) -> List[np.ndarray]:
        """
        Takes a list of representations as input, and returns a list of
        representations as output in the same order. Retrieved metadata
        can be used in this step.
        """
        pass
