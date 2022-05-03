#!/usr/bin/env python3

from __future__ import annotations

import numpy as np

from anndata import AnnData
from scipy.sparse import csr_matrix
from typing import List, Union

from ...utils.anndata_manager import (
    anndata_manager as adm,
    AnnDataKeyIdentifiers,
    _TypePersistLevels,
)


class IsRepresentable:
    """
    A representable object is able to provide a matrix
    representation of an AnnData object, and knows if this
    representation is in the same space and basis as the
    original features space.

    Prameters
    ---------
    repr_key: Union[str, AnnDataKeyIdentifiers]
        String representation that will be used to save
        the matrix representation in AnnData objects.
        IsRepresentable will ensure this representation is unique,
        by appending a suffix if necessary.

    Attributes
    ----------
    is_feature_space: bool
        Whether the matrix representation is in the same features
        space than initial datasets, with the same basis.
    """

    def __init__(self, repr_key: Union[str, AnnDataKeyIdentifiers]):
        # Finds a suffix so that representation is unique
        self.repr_key = repr_key
        self.is_feature_space: bool = True

    def write_representation(
        self,
        adata: AnnData,
        X: Union[np.ndarray, csr_matrix],
        is_feature_space: bool,
        persist: _TypePersistLevels = "pipeline",
    ) -> None:
        """
        Inserts a new representation of a dataset in the .obsm
        field of an AnnData object.

        Parameters
        ----------
        adata: AnnData
            AnnData object to write in

        X: Union[np.ndarray, csr_matrix]
            Representation matrix to write in adata

        is_feature_space: bool
            Matrix representation is expressed in the initial features
            space, with the initial basis.

        persist: Literal["output", "pipeline", "layer"]
            Life duration of the matrix. If "output", won't be erased. If
            "pipeline", will be erased at the end of a Model. If "layer",
            will be erased at the end of the next layer fit().
        """
        self.is_feature_space = is_feature_space
        adm.set_value(
            adata=adata,
            key=self.repr_key,
            field="obsm",
            value=X,
            persist=persist,
        )

    def get_representation(self, adata: AnnData) -> np.ndarray:
        """
        Retrieves matrix representation of an AnnData object that
        has already been written.

        Parameters
        ----------
        adata: AnnData
            AnnData object to retrieve representation from.
        """
        X = adm.get_value(adata=adata, key=self.repr_key)
        assert X is not None, "Representation has not been computed."
        if isinstance(X, csr_matrix):
            X = X.toarray()
        return X

    @staticmethod
    def assert_representation_equals(
        representers: List[IsRepresentable],
        datasets: List[AnnData],
    ) -> None:
        """
        For testing purposes. Tests if a list of IsRepresentable objects
        have similar representations of a set of AnnData datasets.
        """
        if len(representers) == 0:
            return
        for adata in datasets:
            X_ref = representers[0].get_representation(adata)
            for rpr in representers[1:]:
                np.testing.assert_array_almost_equal(
                    X_ref, rpr.get_representation(adata)
                )
