#!/usr/bin/env python3

from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix

import numpy as np
from typing import Dict, List, Tuple
from anndata import AnnData


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

    def __init__(self, metadata_keys: List[str]):
        self.metadata_keys: List[str] = metadata_keys
        self.matchings: Dict[Tuple[int, int], csr_matrix] = {}
        self.n_matchings: int = 0
        self.datasets: List[AnnData] = []  # pointer to AnnDatas
        self.reference_idx: int = -1
        self.n_datasets: int = 0
        self.fitted: bool = False

    def is_referenced(self) -> bool:
        """
        Returns True if the matching uses a reference dataset.
        """
        return self.reference_idx != -1

    def get_dataset_idx(self, adata: AnnData) -> int:
        """
        Internal methods that returns AnnData index in self.datasets.
        Raises a KeyError if the AnnData is not found.

        Parameters
        ----------
        adata: AnnData
            Dataset to seek in self.datasets.
        """
        for i, adata_i in enumerate(self.datasets):
            if adata_i is adata:
                return i
        raise KeyError("AnnData not found in self.datasets.")

    def set_matrix(self, adata: AnnData, dataset_key: str, X: np.ndarray) -> None:
        """
        Registers a matrix in an AnnData object, under a unique string identifier.

        Parameters
        ----------
        adata: AnnData
            Target dataset

        dataset_key: str
            Target matrix identifier

        X: np.ndarray
            Matrix to write
        """
        assert dataset_key not in adata.uns["transmorph"]
        adata.uns["transmorph"][dataset_key] = X

    def get_matrix(self, adata: AnnData, dataset_key: str) -> np.ndarray:
        """
        Retrieves a matrix stored in the AnnData object by set_matrix.

        Parameters
        ----------
        adata: AnnData
            Target dataset

        dataset_key: str
            Target matrix identifier

        Returns
        -------
        The required np.ndarray.
        """
        assert dataset_key in adata.uns["transmorph"]
        return adata.uns["transmorph"][dataset_key]

    def to_match(self, adata: AnnData):
        """
        Retrieves the vectorized preprocessed dataset, to use when implementing
        _match2.

        Parameters
        ----------
        adata: AnnData
            Target dataset
        """
        return self.get_matrix(adata, "_to_match")

    def delete_matrix(self, adata: AnnData, dataset_key: str) -> None:
        """
        Deletes the matrix stored in the AnnData object by set_matrix.

        Parameters
        ----------
        adata: AnnData
            Target dataset

        dataset_key: str
            Target matrix identifier
        """
        assert dataset_key in adata.uns["transmorph"]
        del adata.uns["transmorph"][dataset_key]

    def _check_input(self, adata: AnnData, dataset_key: str = "") -> None:
        """
        Checks if an AnnData is eligible for the given matching. By default, only
        checks if the parameter is a AnnData and if it contains required metadata.
        Can be inherited or overrode when implementing matchings.

        Parameters
        ----------
        adata: AnnData
            AnnData object to check

        dataset_key: str
            String identifier locating the vectorized dataset at this point of
            the pipeline. If "", use adata.X instead.

        Returns
        -------
        Whether $t is a valid TData object for the current matching.
        """
        if type(adata) is not AnnData:
            raise TypeError(f"Error: AnnData expected, found {type(adata)}.")
        if dataset_key != "":  # If "" then use adata.X
            if dataset_key not in adata.uns["transmorph"]:
                raise KeyError(f"Error: missing dataset key {dataset_key}")
        for key in self.metadata_keys:
            if key not in adata.uns["transmorph"]:
                raise KeyError(f"Error: missing metadata {key}.")

    def _preprocess(
        self, adata1: AnnData, adata2: AnnData, dataset_key: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Matching-specific preprocessing step, useful to avoid redundant actions.
        It must return vectorized representations of both datasets. This method can
        also be used to write addtitional information or metadata using set_matrix()
        method if necessary. This method can be left as default.

        Parameters
        ----------
        adata1: AnnData
            Source dataset

        adata2: AnnData
            Reference dataset

        dataset_key: str
            Location of vectorized datasets at this point of the pipeline.
        """
        if dataset_key == "":
            return adata1.X, adata2.X
        return (
            self.get_matrix(adata1, dataset_key),
            self.get_matrix(adata1, dataset_key),
        )

    def _clean(self, adata1: AnnData, adata2: AnnData) -> None:
        """
        Useful to delete matrices stored during _preprocess() function.

        Parameters
        ----------
        adata1: AnnData
            Source dataset

        adata2: AnnData
            Reference dataset
        """
        pass

    @abstractmethod
    def _match2(self, adata1: AnnData, adata2: AnnData) -> csr_matrix:
        """
        Returns a discrete matching T between $t1 and $t2. In this matching,
        T[i,j] is the matching strength between $t1[i] and $t2[j], the higher
        the more probable the correspondence is. There is no requirement such
        that being a probabilistic or boolean matching.

        This method *must* be implemented in every MatchingABC implementation.

        Parameters
        ----------
        adata1: AnnData
            Source dataset

        adata2: AnnData
            Reference dataset

        Returns
        -------
        Sparse matching T of size (#t1, #t2).
        """
        pass

    def get_matching(self, adata1: AnnData, adata2: AnnData) -> csr_matrix:
        """
        Return the matching between two datasets. Throws an error
        if matching is not fitted, or if the required matching does not exist.

        Parameters
        ----------
        adata1: AnnData
            Source dataset.

        adata2: AnnData
            Reference dataset

        Returns
        -------
        T = (adata1.n_obs, adata2.n_obs) sparse array, where Tkl is the
        matching strength between adata1_k and adata2_l.
        """
        i1 = self.get_dataset_idx(adata1)
        i2 = self.get_dataset_idx(adata2)
        matching = self.matchings.get((i1, i2), None)
        if matching is None:
            matching = self.matchings.get((i2, i1), None)
            if matching is None:
                raise ValueError("No matching between the AnnDatas.")
            matching = csr_matrix(matching.T)
        return matching

    def fit(
        self,
        datasets: List[AnnData],
        dataset_key: str = "",
        reference: AnnData = None,
    ) -> None:
        """
        Computes the matching between a set of TData. Should not be overrode in
        the implementation in order to ensure compatibility between Matching and
        all Mergings.

        Parameters:
        -----------
        datasets: List[AnnData]
            List of datasets.

        dataset_key: str, default = ""
            Dictionary key, locating where preprocessed vectorized datasets are.

        reference: TData, default = None
            Optional reference dataset. If left empty, all $datasets are matched
            between one another.
        """
        # Checking all datasets are correct
        for dataset in self.datasets:
            self._check_input(dataset, dataset_key)

        self.n_datasets = len(self.datasets)

        # Identifying the reference dataset, then storing it
        ref_idx = -1
        if reference is not None:
            for i, adata in enumerate(datasets):
                if adata is reference:
                    ref_idx = i
                    break
            if ref_idx == -1:
                raise ValueError("Reference not found in datasets.")
            self.datasets = [reference] + datasets
        else:
            self.datasets = datasets
        self.reference_idx = ref_idx

        # Computing the pairwise matchings
        self.fitted = False
        ref_datasets = [reference] if reference is not None else datasets
        ref_idx = [ref_idx] if reference is not None else np.arange(self.n_datasets)
        self.matchings = {}
        for i, src in enumerate(datasets):
            for j, ref in zip(ref_idx, ref_datasets):
                if i == j or (j, i) in self.matchings:
                    continue
                Xi, Xj = self._preprocess(src, ref, dataset_key)
                self.set_matrix(src, "_to_match", Xi)
                self.set_matrix(ref, "_to_match", Xj)
                self.matchings[i, j] = self._match2(src, ref)
                self.delete_matrix(src, "_to_match")
                self.delete_matrix(ref, "_to_match")
                self._clean(src, ref)
        self.n_matchings = len(self.matchings)
        self.fitted = True
