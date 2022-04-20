#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from transmorph.engine.profiler import IsProfilable
from transmorph.engine.traits import CanLog


class Preprocessing(ABC, CanLog, IsProfilable):
    """
    Abstract class for Preprocessing objects. Implements a method
    transform(List[np.ndarray]) that computes the preprocessing.
    Child classes can be enriched by traits.
    """

    def __init__(self, preserves_space: bool = False, str_identifier: str = "DEFAULT"):
        CanLog.__init__(self, str_identifier=f"PREPROCESSING_{str_identifier}")
        self.preserves_space = preserves_space

    @abstractmethod
    def transform(self, datasets: List[np.ndarray]) -> List[np.ndarray]:
        """
        Takes a list of representations as input, and returns a list of
        representations as output in the same order. Retrieved metadata
        can be used in this step.
        """
        pass
