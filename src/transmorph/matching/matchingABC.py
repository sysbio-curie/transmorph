#!/usr/bin/env python3

from abc import ABC, abstractmethod
from logging import warn
from scipy.sparse import csr_matrix

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from anndata import AnnData

from ..subsampling.subsamplingABC import SubsamplingABC
from ..subsampling import SubsamplingKeepAll
from ..utils import anndata_interface as ad


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
    metadata_keys: List[str], default = []
        TData.metadata keywords needed by the matching method.

    subsampling: SubsamplingABC, default = None
        Subsampling scheme to apply before computing the matching,
        can be very helpful when dealing with large datasets.

    Attributes
    ----------
    datasets: List[AnnData]
        Datasets the matching has been fitted with.

    fitted: boolean
        Is true if match() method has been successfully executed.

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

    reference_idx: int
        Reference dataset index.
    """

    def __init__(
        self,
        metadata_keys: List[str],
        subsampling: Optional[SubsamplingABC] = None,
    ):
        self.metadata_keys: List[str] = metadata_keys
        if subsampling is None:
            subsampling = SubsamplingKeepAll()
        self.subsampling = subsampling
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
        Internal method that returns AnnData index in self.datasets.
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

    @staticmethod
    def to_match(adata: AnnData) -> np.ndarray:
        """
        Retrieves the vectorized preprocessed dataset, to use when implementing
        _match2.

        Parameters
        ----------
        adata: AnnData
            Target dataset
        """
        return ad.get_matrix(adata, "_to_match")

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
            if not ad.isset_matrix(adata, dataset_key):
                raise KeyError(f"Error: missing dataset key {dataset_key}")
        for key in self.metadata_keys:
            if not ad.isset_info(adata, key):
                raise KeyError(f"Error: missing metadata {key}.")
        if adata.n_obs > 10000 and type(self.subsampling) is SubsamplingKeepAll:
            warn(
                "Large dataset detected. You may want to consider a subsampling"
                " strategy to improve performance and facilitate convergence "
                " (e.g. SubsamplingVertexCover)."
            )

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
            ad.get_matrix(adata1, dataset_key),
            ad.get_matrix(adata2, dataset_key),
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
    def _match2(
        self, adata1: AnnData, adata2: AnnData
    ) -> Union[np.ndarray, csr_matrix]:
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

    def get_matching(
        self, adata1: AnnData, adata2: AnnData, row_normalize: bool = False
    ) -> csr_matrix:
        """
        Return the matching between two datasets. Throws an error
        if matching is not fitted, or if the required matching does not exist.

        Parameters
        ----------
        adata1: AnnData
            Source dataset.

        adata2: AnnData
            Reference dataset

        TODO: restore normalization?

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
                raise ValueError("No matching found between the AnnDatas.")
            matching = csr_matrix(matching.T)
        if row_normalize:
            coefs = np.array(matching.sum(axis=1))
            coefs[coefs == 0.0] = 1.0
            matching = csr_matrix(matching / coefs)
        return matching

    def get_anchors(self, adata: AnnData) -> np.ndarray:
        """
        Returns a boolean vector containing sample indices used during
        matching, zero indices representing ignored points.
        """
        return self.subsampling.get_anchors(adata)

    def fit(
        self,
        datasets: List[AnnData],
        dataset_key: str = "",
        reference: AnnData = None,
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
        # Checking all datasets are correct
        for dataset in datasets:
            self._check_input(dataset, dataset_key)

        self.n_datasets = len(datasets)

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

        # Computing subsampling if necessary
        self.subsampling.subsample(datasets, X_kw=dataset_key)

        # Computing the pairwise matchings
        self.fitted = False
        ref_datasets = [reference] if reference is not None else datasets
        ref_indices = [ref_idx] if reference is not None else np.arange(self.n_datasets)
        self.matchings = {}
        for i, src in enumerate(datasets):
            for j, ref in zip(ref_indices, ref_datasets):
                if i == j or (j, i) in self.matchings:
                    continue
                Xi, Xj = self._preprocess(src, ref, dataset_key)
                anci, ancj = (
                    self.subsampling.get_anchors(src),
                    self.subsampling.get_anchors(ref),
                )
                Xi = Xi[anci]  # Points to match
                Xj = Xj[ancj]
                ad.set_matrix(src, "_to_match", Xi)
                ad.set_matrix(ref, "_to_match", Xj)
                T = self._match2(src, ref)
                if type(T) is np.ndarray:
                    T = csr_matrix(T)
                assert type(T) is csr_matrix
                # TODO: Extrapolate matching to non-anchor points
                # For now, just ignore unmatched points
                ni, nj = src.n_obs, ref.n_obs
                rows, cols, data = [], [], []
                T = T.tocoo()
                anc_to_ind_i = np.arange(ni)[anci]
                anc_to_ind_j = np.arange(nj)[ancj]
                for k, l, v in zip(T.row, T.col, T.data):
                    rows.append(anc_to_ind_i[k])
                    cols.append(anc_to_ind_j[l])
                    data.append(v)
                self.matchings[i, j] = csr_matrix((data, (rows, cols)), shape=(ni, nj))
                ad.delete_matrix(src, "_to_match")
                ad.delete_matrix(ref, "_to_match")
                self._clean(src, ref)
        self.n_matchings = len(self.matchings)
        self.fitted = True
