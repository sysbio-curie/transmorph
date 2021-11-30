#!/usr/bin/env python3

from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix

import numpy as np
from typing import Union, List


class MatchingABC(ABC):
    """
    A matching is a class containing a function match(x1, ..., xn), able
    to predict matching between dataset samples (possibly fuzzy). Any class
    implementing matching must implement a

        _match2(self, x1: np.ndarray, x2: np.ndarray)

    method, returning a possibly sparse T = (x1.shape[0], x2.shape[0]) array
    where T[i, j] = prob(x1_i matches x2_j).

    Parameters
    ----------
    use_sparse: boolean, default = True
        Save matchings as sparse matrices.

    use_reference: int, default = -1
        When matching, use the dataset i as the reference for matching. In this
        case, self.matchings will contain n - 1 matchings, where matching[k] is
        the matching between k and i if k < i and between k + 1 and i if k > i.

    Attributes
    ----------
    fitted: boolean
        Is true if match() method has been successfully exectuted.

    matchings: list of arrays
        After calling match(x0, ..., xn), matching[i(i-1)/2+j] contains the
        matching between xi and xj (with i > j).
    """

    def __init__(
        self,
        use_sparse: bool = True,
        use_reference: bool = False,
        reference: np.ndarray = None,
    ):
        self.fitted = False
        self.matchings = []
        self.n_matchings = 0
        self.use_sparse = use_sparse
        self.use_reference = use_reference
        self.reference = reference

    @abstractmethod
    def _match2(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        pass

    def get(
        self, i: int, j: int = -1, normalize: bool = False
    ) -> Union[np.ndarray, csr_matrix]:
        """
        Return the matching between datasets i and j. Throws an error
        if matching is not fitted, or if i == j.

        Parameters
        ----------
        i: int
            Index of the source dataset (samples in rows).

        j: int
            Index of the reference dataset (samples in columns), useless
            if self.use_reference = True.

        normalize: bool
            Normalize each non-zero row to sum up to one.

        Returns
        -------
        T = (xi.shape[0], xj.shape[0]) sparse array, where Tkl is the
        matching strength between xik and xjl (or reference).
        """
        assert self.fitted, "Error: matching not fitted, call match() first."
        assert i != j, "Error: i = j."
        transpose = i < j
        if transpose:
            i, j = j, i
        if self.use_reference:
            assert j == -1, "Error: impossible to set j when use_reference=True."
            index = i
        else:
            index = int(i * (i - 1) / 2 + j)
        assert index < len(self.matchings), f"Index ({i}, {j}) is out of bounds."
        T = self.matchings[index]
        if transpose:
            if type(T) == np.ndarray:
                T = T.T
            elif type(T) == csr_matrix:
                T = csr_matrix(T.transpose())  # Cast necessary (avoids CSC)
            else:
                raise NotImplementedError
        if normalize:
            normalizer = T.sum(axis=1)
            normalizer[normalizer == 0.0] = 1.0
            return T / normalizer
        return T

    def match(self, *datasets: np.ndarray) -> List[np.ndarray]:
        """
        Matches all pairs of different datasets together. Returns results
        in a dictionary, where d[i,j] is the matching between datasets i
        and j represented as a (ni, nj) numpy array -- possibly fuzzy.

        Parameters:
        -----------
        *datasets: list of datasets
            List of at least two datasets.
        """
        self.fitted = False
        self.matchings = []
        nd = len(datasets)
        assert nd > 1, "Error: at least 2 datasets required."
        for i in range(nd):
            di = datasets[i]
            for j in range(i):
                matching = self._match2(di, datasets[j])
                if self.use_sparse:
                    matching = csr_matrix(matching)
                self.matchings.append(matching)
        self.n_matchings = len(self.matchings)
        self.fitted = True
        return self.matchings
