#!/usr/bin/env python3

import numpy as np

from abc import ABC, abstractmethod
from typing import List

from ..traits.canlog import CanLog
from ..traits.isprofilable import IsProfilable


class Checking(ABC, CanLog, IsProfilable):
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
        Threshold value to compare statistical test result with. ByCheckingEntropy
        default, checking passes if test value is below the threshold.

    accept_if_lower: bool, default = True
        If True, then lower is better. Otherwise, higher is better.

    Attributes
    ----------
    last_value: float
        Value returned by the last call to the testing algorithm. Can
        be retrived by other modules for logging purposes for instance.
    """

    def __init__(self, str_identifier: str = "DEFAULT"):
        CanLog.__init__(self, str_identifier=f"CHECKING_{str_identifier}")
        IsProfilable.__init__(self)

    def check_input(self, datasets: List[np.ndarray]) -> None:
        """
        Checks input validity, can be overriden at the condition to
        start by calling the base verify_input inside.
        """
        pass

    @abstractmethod
    def check(self, datasets: List[np.ndarray]) -> bool:
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
        pass
