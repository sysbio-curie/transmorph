#!/usr/bin/env python3

from __future__ import annotations

from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix
from transmorph.engine.profiler import IsProfilable, profile_method
from transmorph.engine.traits import CanLog

import numpy as np
from typing import Dict, List, Literal, Tuple


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
        self.matchings: Dict[Tuple[int, int], csr_matrix] = {}

    def check_input(self, datasets: List[np.ndarray]) -> None:
        """
        Takes a list of np.ndarray representing datasets to match,
        and verifies their validity. Should raise warnings or
        errors in case of unexpected input. Will be called before
        carrying out the matching task. Can be overrode by child
        classes.
        """
        pass

    def get_matching(
        self,
        ind_1: int,
        ind_2: int,
        mode: Literal["raw", "boolean", "row_normalized"] = "row_normalized",
    ) -> csr_matrix:
        """
        Return the matching between two datasets. Throws an error
        if matching is not fitted, or if the required matching does not exist.

        Parameters
        ----------
        ind_1: ind
            Index of the first dataset

        ind_2: ind
            Index of the second dataset

        mode: Literal["raw", "boolean", "row_normalized"],
              default = "row_normalized"
            Transformation to apply to the matching matrix
            - "raw": Return the matching without modifications
            - "boolean": Return a boolean version of the matching,
              where all nonzero entries are set to True
            - "raw_normalized": Return a row normalized version of
              the matching, where all nonzero rows marginalize to 1.

        Returns
        -------
        T = (n_ind1, n_ind2) CSR sparse matrix, where Tkl is the
        matching strength between X_ind1_k and X_ind2_l.
        """
        matching = self.matchings.get((ind_1, ind_2), None)
        if matching is None:
            matching = self.matchings.get((ind_2, ind_1), None)
            if matching is None:
                self.raise_error(
                    ValueError,
                    f"No matching found between indices {ind_1} and {ind_2}.",
                )
            matching = csr_matrix(matching.T)
        if mode == "row_normalize":
            coefs = np.array(matching.sum(axis=1))
            coefs[coefs == 0.0] = 1.0
            return csr_matrix(matching / coefs)
        elif mode == "boolean":
            return matching.astype(bool)
        elif mode == "raw":
            return matching
        else:
            self.raise_error(ValueError, f"Unrecognized mode {mode}.")

    @abstractmethod
    @profile_method
    def fit(
        self,
        datasets: List[np.ndarray],
        reference_idx: int = -1,
    ) -> None:
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
