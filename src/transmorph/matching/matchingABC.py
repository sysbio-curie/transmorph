#!/usr/bin/env python3

from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix

import numpy as np
from typing import Iterable, Tuple, Union, List
import scanpy as sc


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
    use_reference: int, default = -1
        When matching, use the dataset i as the reference for matching. In this
        case, self.matchings will contain n - 1 matchings, where matching[k] is
        the matching between k and i if k < i and between k + 1 and i if k > i.

    metadata_needed: List[str], default = []
        TData.metadata keywords needed by the matching method.

    Attributes
    ----------
    datasets: List[TData]
        Datasets the matching has been fitted with.

    fitted: boolean
        Is true if match() method has been successfully exectuted.

    n_datasets: int
        Number of TData used during fitting.

    n_matchings: int
        Number of matchings between datasets. Is equal to n_datasets - 1 if
        matching is referenced, (2 choose n_datasets) otherwise.

    matchings: list of arrays
        After calling match(x0, ..., xn), list of matchings. If matching is
        referenced, matching[i] contains matching between datasets[i] and
        reference dataset. Otherwise, matching[i(i-1)/2+j] contains the
        matching between xi and xj (with i > j).

    reference: TData
        If matching is referenced, contains a link to the reference TData.
    """

    def __init__(
        self,
        use_sparse: bool = True,
        metadata_needed: List[str] = [],
    ):
        self.datasets = []
        self.fitted = False
        self.n_datasets = 0
        self.n_matchings = 0
        self.matchings = []
        self.metadata_needed = metadata_needed
        self.reference = None
        self.use_reference = False

    def _check_input(self, adata: sc.AnnData) -> bool:
        """
        Checks if a TData is eligible for the given matching. By default, only
        checks if the parameter is a TData and if it contains required metadata.
        Can be inherited or overrode when implementing matchings.

        Parameters
        ----------
        t: TData
            TData object to check

        Returns
        -------
        Whether $t is a valid TData object for the current matching.
        """
        if type(adata) is not sc.AnnData:
            print("Error: sc.AnnData expected.")
            return False
        for key in self.metadata_needed:
            if key not in adata.uns["_transmorph"]["matching"]:
                print(f"Error: missing metadata {key}.")
                return False
        return True

    def _preprocess(
        self, adata1: sc.AnnData, adata2: sc.AnnData
    ) -> Tuple[sc.AnnData, sc.AnnData]:
        """
        Preprocessing pipeline of pairs of datasets prior to matching. By
        default, identity mapping. Can be overrode when implementing matchings.

        Parameters
        ----------
        adata1: TData
            Source dataset in the matching.

        adata2: TData
            Reference dataset in the matching.

        Returns
        -------
        Preprocessed representations of $t1 and $t2.
        """
        return adata1, adata2  # Default: identity mapping

    @abstractmethod
    def _match2(
        self, adata1: sc.AnnData, adata2: sc.AnnData
    ) -> Union[csr_matrix, np.ndarray]:
        """
        Returns a discrete matching T between $t1 and $t2. In this matching,
        T[i,j] is the matching strength between $t1[i] and $t2[j], the higher
        the more probable the correspondence is. There is no requirement such
        that being a probabilistic or boolean matching.

        This method *must* be implemented in every MatchingABC implementation.

        Parameters
        ----------
        adata1: TData
            Source dataset

        adata2: TData
            Reference dataset

        Returns
        -------
        Possibly sparse matching T of size #t1, #t2.
        """
        pass

    def iter_datasets(self) -> Iterable[sc.AnnData]:
        """
        Use this iterator to iterate over datasets.
        """
        for dataset in self.datasets:
            yield dataset

    def get_dataset(self, i: int) -> sc.AnnData:
        """
        Returns dataset at the i-th position.
        """
        assert i < self.n_datasets, "Error: dataset index out of bounds."
        return self.datasets[i]

    def get_matching(self, i: int, j: int = -1, normalize: bool = False) -> csr_matrix:
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
        transpose = j != -1 and j < i  # We only store matchings for i < j
        if transpose:
            i, j = j, i
        if self.use_reference:
            assert j == -1, "Error: impossible to set j when use_reference=True."
            index = i
        else:
            index = int(j * (j - 1) / 2 + i)
        assert index < len(self.matchings), f"Index ({i}, {j}) is out of bounds."
        T = self.matchings[index]
        if transpose:
            T = csr_matrix(T.transpose())  # Cast necessary (avoids CSC)
        if normalize:
            normalizer = T.sum(axis=1)
            normalizer[normalizer == 0.0] = 1.0
            T = csr_matrix(T / normalizer)  # / returns a np.matrix
        return T

    def fit(
        self,
        datasets: List[sc.AnnData],
        reference: sc.AnnData = None,
    ) -> List[np.ndarray]:
        """
        Computes the matching between a set of TData. Should not be overrode in
        the implementation in order to ensure compatibility between Matching and
        all Mergings.

        Parameters:
        -----------
        *datasets: List[TData]
            List of datasets.

        reference: TData, default = None
            Optional reference dataset. If left empty, all $datasets are matched
            between one another.
        """
        if reference is not None:  # By convention, reference dataset is put on head
            self.reference = reference
            self.datasets = [reference] + datasets
        else:
            self.datasets = datasets
        for dataset in self.datasets:
            assert self._check_input(dataset)
        self.n_datasets = len(self.datasets)
        self.reference = reference
        self.use_reference = reference is not None
        self.fitted = False
        self.matchings = []
        nd = len(self.datasets)
        if reference is not None:
            assert nd > 1, "Error: at least 1 dataset required."
            for di in self.datasets:
                adata1, adata2 = self._preprocess(di, reference)
                matching = self._match2(adata1, adata2)
                if type(matching) is np.ndarray:
                    matching = csr_matrix(matching, shape=matching.shape)
                self.matchings.append(matching)
        else:
            assert nd > 1, "Error: at least 2 datasets required."
            for j, dj in enumerate(self.datasets):
                for di in self.datasets[:j]:
                    adata1, adata2 = self._preprocess(di, dj)
                    matching = self._match2(adata1, adata2)
                    if type(matching) is np.ndarray:
                        matching = csr_matrix(matching, shape=matching.shape)
                    self.matchings.append(matching)
        self.n_matchings = len(self.matchings)
        self.fitted = True
        return self.matchings
