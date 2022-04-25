#!/usr/bin/env python3

import numpy as np

from anndata import AnnData
from typing import Callable, List, Optional

from .hasmetadata import HasMetadata
from ..subsampling import Subsampling, KeepAll
from ...utils import anndata_manager as adm, AnnDataKeyIdentifiers


class IsSubsamplable:
    """
    This trait allows a class to host a subsampling algorithm.
    """

    def __init__(self, subsampling: Optional[Subsampling] = None) -> None:
        if subsampling is None:
            subsampling = KeepAll()
        self.subsampling = subsampling

    def subsample(
        self,
        datasets: List[AnnData],
        matrices: List[np.ndarray],
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
        if log_callback is not None:
            log_callback(f"Applying subsampling {self.subsampling}.")
        if isinstance(self.subsampling, HasMetadata):
            self.subsampling.retrieve_all_metadata(datasets)
        to_compute = []
        for i, adata in enumerate(datasets):
            anchors = adm.get_value(adata, AnnDataKeyIdentifiers.SubsamplingAnchors)
            references = adm.get_value(
                adata, AnnDataKeyIdentifiers.SubsamplingReferences
            )
            if anchors is None or references is None:
                # Needs to be computed
                to_compute.append((i, matrices[i]))
        subsampling_results = self.subsampling.subsample([mtx for _, mtx in to_compute])
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

    def slice_matrices(
        self,
        datasets: List[AnnData],
        matrices: List[np.ndarray],
        pairwise: bool = False,
    ) -> List[np.ndarray]:
        """
        Returns row-sliced views of input matrices according to
        the subsampling computed.

        Parameters
        ----------
        datasets: List[AnnData]
            AnnData objects containing subsampling data.

        matrices: List[np.ndarray]
            Arrays to slice in a list of same length as datasets.
        """
        result = []
        for adata, X in zip(datasets, matrices):
            anchors = IsSubsamplable.get_anchors(adata)
            X_sliced = X[anchors]
            if pairwise:
                X_sliced = X[:, anchors]
            result.append(X_sliced)
        return result
