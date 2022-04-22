#!/usr/bin/env python3

import numpy as np

from anndata import AnnData
from scipy.sparse import csr_matrix
from typing import Union

from ...utils import anndata_manager as adm, AnnDataKeyIdentifiers


class IsRepresentable:
    """
    A representable object is able to provide a matrix
    representation of an AnnData object. It keeps track whether
    its representation is in the initial feature space.
    """

    def __init__(self, repr_key: Union[str, AnnDataKeyIdentifiers]):
        self.repr_key = repr_key
        self.is_feature_space = True

    def write_representation(
        self,
        adata: AnnData,
        X: Union[np.ndarray, csr_matrix],
        is_feature_space: bool,
    ) -> None:
        """
        Inserts a new AnnData representation.
        """
        self.is_feature_space = is_feature_space
        adm.set_value(
            adata=adata,
            key=self.repr_key,
            field="obsm",
            value=X,
            persist="pipeline",
        )

    def get_representation(self, adata: AnnData) -> np.ndarray:
        """
        Returns a matrix view of a given AnnData.
        """
        X = adm.get_value(adata=adata, key=self.repr_key)
        assert X is not None, "Representation has not been computed."
        if isinstance(X, csr_matrix):
            X = X.toarray()
        return X
