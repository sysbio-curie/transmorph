#!/usr/bin/env python3

import numba
import numpy as np
import sccover

from anndata import AnnData
from scipy.sparse import csr_matrix
from typing import Callable, List, Optional, Union

from ..subsampling import Subsampling, KeepAll
from ..traits.utils import preprocess_traits
from ...utils.anndata_manager import anndata_manager as adm, AnnDataKeyIdentifiers


@numba.njit
def _get_cell_anchor_matrix_njit(nx, na, anchors, connectivities):
    T = np.zeros((nx, na))
    for i in range(nx):
        for anchor_ind in range(na):
            T[i, anchor_ind] = connectivities[i, anchors[anchor_ind]]
    return T


def get_cell_anchor_matrix(adata):
    subsampling_set_key = adm.gen_keystring(AnnDataKeyIdentifiers.SubsamplingAnchors)
    sccover.get_closest_anchor(adata, key_set=subsampling_set_key)
    nx = adata.n_obs
    anchors = np.arange(nx)[adata.obs[subsampling_set_key].astype(bool)]
    na = anchors.shape[0]
    T = _get_cell_anchor_matrix_njit(
        nx, na, anchors, adata.obsp["connectivities"].toarray()
    )
    T_norm = T.sum(axis=1)
    T_norm[T_norm == 0.0] = 1.0
    return T / T_norm[:, None]


class UsesSubsampling:
    """
    This trait allows a class to host a subsampling algorithm, and manipulate
    subsampled matrices. This is a key trait for performance-hungry components.
    """

    def __init__(
        self,
        subsampling: Optional[Subsampling] = None,
    ) -> None:
        if subsampling is None:
            subsampling = KeepAll()
        self.is_subsampled = not isinstance(subsampling, KeepAll)
        self.subsampling = subsampling
        self.references: List[np.ndarray] = []

    @property
    def has_subsampling(self) -> bool:
        return not isinstance(self.subsampling, KeepAll)

    def compute_subsampling(
        self,
        datasets: List[AnnData],
        matrices: List[np.ndarray],
        is_feature_space: bool,
        log_callback: Optional[Callable] = None,
    ) -> None:
        """
        Runs the subsampling on a list of datasets, and
        stores the result in AnnData objects. If subsampling results
        are already found, skips the dataset.

        Parameters
        ----------
        datasets: List[AnnData]
            AnnData objects containing datasets to subsample.

        matrices: List[np.ndarray]
            Matrix representation of datasets to subsample.
        """
        if self.has_subsampling is False:
            return
        if log_callback is not None:
            log_callback(f"Applying subsampling {self.subsampling}.")
        preprocess_traits(self.subsampling, datasets, is_feature_space)
        ndatasets = len(datasets)
        assert len(matrices) == ndatasets, "Inconsistent number of matrices/annadatas."
        to_compute = []
        self.anchors = [None] * ndatasets
        self.references = [None] * ndatasets
        for i, adata in enumerate(datasets):
            anchors = adm.get_value(adata, AnnDataKeyIdentifiers.SubsamplingAnchors)
            references = adm.get_value(
                adata, AnnDataKeyIdentifiers.SubsamplingReferences
            )
            if anchors is None or references is None:
                # Needs to be computed
                to_compute.append((i, matrices[i]))
            else:
                self.anchors[i] = anchors
                self.references[i] = references

        # Computes missing subsamplings
        subsampling_results = self.subsampling.subsample(
            datasets, [mtx for _, mtx in to_compute]
        )
        # We store results in AnnData objects
        for (i, _), (anchors, references) in zip(to_compute, subsampling_results):
            adm.set_value(
                adata=datasets[i],
                key=AnnDataKeyIdentifiers.SubsamplingAnchors,
                field="obs",
                value=anchors,
                persist="output",
            )
            adm.set_value(
                adata=datasets[i],
                key=AnnDataKeyIdentifiers.SubsamplingReferences,
                field="obs",
                value=references,
                persist="output",
            )
            self.anchors[i] = anchors
            self.references[i] = references

        if log_callback is not None:
            for i, anchors in enumerate(self.anchors):
                log_callback(
                    f"Dataset {i} subsampled from {anchors.shape[0]} to "
                    f"{anchors.sum()} samples."
                )

    @staticmethod
    def get_anchors(adata: AnnData) -> np.ndarray:
        """
        Returns the chosen subsample as a boolean numpy array.
        """
        anchors = adm.get_value(adata, AnnDataKeyIdentifiers.SubsamplingAnchors)
        if anchors is None:
            raise KeyError(f"No anchors found for the AnnData {adata}.")
        return anchors

    @staticmethod
    def get_references(adata: AnnData) -> np.ndarray:
        """
        Returns the chosen subsample as a boolean numpy array.
        """
        references = adm.get_value(adata, AnnDataKeyIdentifiers.SubsamplingReferences)
        if references is None:
            raise KeyError(f"No references found for the AnnData {adata}.")
        return references

    def subsample_matrix(
        self,
        matrix: Union[np.ndarray, csr_matrix],
        idx: int,
        idx_2: Optional[int] = None,
        pairwise: bool = False,
    ) -> Union[np.ndarray, csr_matrix]:
        """
        Returns row-sliced matrix with respect to computed subsample.

        Parameters
        ----------
        matrix: np.ndarray
            Array to slice.

        index: int
            Matrix index used in compute_subsampling

        pairwise: bool, default = False
            Also column-slice the matrix, matrix must be squared.
        """
        if self.has_subsampling is False:
            return matrix
        assert idx < len(
            self.anchors
        ), "Ensure compute_subsampling has been called first."
        assert not (idx_2 is not None and pairwise), "Incompatible pairwise non-square."
        anchors = self.anchors[idx]
        X = matrix[anchors]
        if pairwise:
            X = X[:, anchors]
        if idx_2 is not None:
            X = X[:, self.anchors[idx_2]]
        return X

    def subsample_matrices(
        self,
        matrices: List[Union[np.ndarray, csr_matrix]],
        pairwise: bool = False,
    ) -> List[Union[np.ndarray, csr_matrix]]:
        """
        Returns row-sliced views of input matrices according to
        the subsampling computed.

        Parameters
        ----------
        matrices: List[np.ndarray]
            Arrays to slice in a list.

        pairwise: bool, default = False
            Also column-slice the matrix, matrix must be squared.
        """
        result = []
        for i, X in enumerate(matrices):
            result.append(self.subsample_matrix(X, i, pairwise=pairwise))
        return result

    def unsubsample_matrix_exact(
        self,
        matrix: Union[csr_matrix, np.ndarray],
        idx: int,
        idx_2: Optional[int] = None,
    ) -> Union[csr_matrix, np.ndarray]:
        """
        Restores initial structure of a subsampled matrix, possibly
        pairwise. Useful to reverse subsampling of a distance matrix for
        instance.

        Parameters
        ----------
        matrix: Union[csr_matrix, np.ndarray]
            Subsampled matrix to turn back to full size.

        idx: int
            Index of the subsampling to use.

        idx_2: Optional[int]
            In case of pairwise matrix between different datasets, subsampling
            index of the second dataset.
        """
        # Sanity check
        if self.has_subsampling is False:
            return matrix
        assert isinstance(
            matrix, (csr_matrix, np.ndarray)
        ), f"Unknown type: {type(matrix)}"

        # Inverting subsampling indices
        n_samples = self.anchors[idx].shape[0]
        n_features = n_samples
        s_to_S1 = np.arange(n_samples)[self.anchors[idx].astype(bool)]
        s_to_S2 = s_to_S1
        if idx_2 is not None:
            anchors_2 = self.anchors[idx_2]
            n_features = anchors_2.shape[0]
            s_to_S2 = np.arange(n_features)[anchors_2.astype(bool)]

        # Reversing subsampling
        if isinstance(matrix, csr_matrix):
            X_coo = matrix.tocoo()
            srow, scol, sdata = X_coo.row, X_coo.col, X_coo.data
            Srow = s_to_S1[srow]
            if idx_2 is None:
                Scol = scol
            else:
                Scol = s_to_S2[scol]
            return csr_matrix(
                (sdata, (Srow, Scol)),
                shape=(n_samples, n_features),
            )
        else:  # np.ndarray
            X = np.zeros((n_samples, n_features), dtype=matrix.dtype)
            for idx_i, coord_i in enumerate(s_to_S1):
                for idx_j, coord_j in enumerate(s_to_S2):
                    X[coord_i, coord_j] = matrix[idx_i, idx_j]
            return X

    def unsubsample_matrix_transitive(
        self,
        matrix: csr_matrix,
        adata_src: AnnData,
        adata_ref: AnnData,
    ) -> csr_matrix:
        """
        TODO
        """
        Tx = get_cell_anchor_matrix(adata_src)
        Ty = get_cell_anchor_matrix(adata_ref)
        T_tot = Tx @ matrix @ Ty.T
        T_tot_norm = T_tot.sum(axis=1)
        T_tot_norm[T_tot_norm == 0.0] = 1.0
        return csr_matrix(T_tot / T_tot_norm[:, None])
