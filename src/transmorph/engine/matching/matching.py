#!/usr/bin/env python3

from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod
from typing import List

from . import _TypeMatchingSet
from ..traits import CanLog, IsProfilable


class Matching(ABC, IsProfilable, CanLog):
    """
    A matching is a class containing a function match(x1, ..., xn), able
    to predict matching between dataset samples (possibly fuzzy). Any class
    implementing matching must implement a

        _match2(self, x1: np.ndarray, x2: np.ndarray)

    method, returning a possibly sparse T = (x1.shape[0], x2.shape[0]) array
    where T[i, j] = prob(x1_i matches x2_j).

    Parameters
    ----------
    subsampling: SubsamplingABC, default = None
        Subsampling scheme to apply before computing the matching,
        can be very helpful when dealing with large datasets.

    str_identifier: str
        String representation of the matching algorithm. Will
        typically be the matching algorithm name. For logging purposes.

    Attributes
    ----------
    matchings: list of arrays
        After calling match(x0, ..., xn), list of matchings. If matching is
        referenced, matching[i] contains matching between datasets[i] and
        reference dataset. Otherwise, matching[i(i-1)/2+j] contains the
        matching between xi and xj (with i > j).
    """

    def __init__(
        self,
        str_identifier: str = "DEFAULT",
    ):
        CanLog.__init__(self, str_identifier=f"MATCHING_{str_identifier}")

    def check_input(self, datasets: List[np.ndarray]) -> None:
        """
        Takes a list of np.ndarray representing datasets to match,
        and verifies their validity. Should raise warnings or
        errors in case of unexpected input. Will be called before
        carrying out the matching task. Can be overrode by child
        classes.
        """
        pass

    @abstractmethod
    def fit(
        self,
        datasets: List[np.ndarray],
        reference_idx: int = -1,
    ) -> _TypeMatchingSet:
        """
        Computes the matching between a set of AnnData. Should not be overriden in
        the implementation in order to ensure compatibility between Matching and
        all Merging.

        Parameters:
        -----------
        datasets: List[AnnData]
            List of datasets.

        dataset_key: str, default = ""
            Dictionary key, locating where preprocessed vectorized datasets are.

        reference: AnnData, default = None
            Optional reference dataset. If left empty, all $datasets are matched
            between one another.
        """
        pass
