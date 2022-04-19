#!/usr/bin/env python3

from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod
from anndata import AnnData
from scipy.sparse import csr_matrix
from transmorph import logger
from transmorph.utils import anndata_manager as adm
from transmorph.utils import AnnDataKeyIdentifiers
from typing import Any, Dict, Hashable, List, Optional, Tuple, Type, Union

from ..preprocessing.preprocessingABC import PreprocessingABC


# A trait is a small module of features that can be added
# to a class using inheritance. It allows code factorization,
# and easier compatibility checking. The base trait does nothing
# but checking if an object is endowed with it.


def assert_trait(obj: Any, traits: Union[Type, Tuple[Type, ...]]):
    """
    Raises an exception if $obj is not endowed with the
    trait $trait.
    """
    if isinstance(obj, traits):
        return
    if isinstance(traits, Type):
        all_traits: str = traits.__name__
    else:
        all_traits: str = ", ".join([trait.__name__ for trait in traits])
    raise TypeError(
        f"Object {obj} of type {type(obj)} is not endowed"
        f" with trait(s) {all_traits}."
    )


class IsPreprocessable:
    """
    A preprocessable object is a layer that can contain internal
    preprocessing steps.
    """

    def __init__(self) -> None:
        self.preprocessings: List[PreprocessingABC] = []

    @property
    def has_preprocessings(self) -> bool:
        return len(self.preprocessings) > 0

    def add_preprocessing(self, preprocessing: PreprocessingABC) -> None:
        """
        Adds a preprocessing step to the layer, that will be applied
        before running the internal algorithm.
        """
        self.preprocessings.append(preprocessing)

    def preprocess(
        self, datasets: List[AnnData], representer: IsRepresentable
    ) -> List[np.ndarray]:
        """
        Runs all preprocessings.
        """
        IsRepresentable.assert_valid(representer)
        Xs = [representer.get(adata) for adata in datasets]
        for preprocessing in self.preprocessings:
            # If necessary, we let preprocessing retrieve
            # additional information
            if isinstance(preprocessing, HasMetadata):
                preprocessing.retrieve_all_metadata(datasets)
            Xs = preprocessing.transform(Xs)
        return Xs


class IsRepresentable:
    """
    A representable object is able to provide a matrix
    representation of an AnnData object.
    """

    def __init__(self, repr_key: Union[str, AnnDataKeyIdentifiers]):
        self.repr_key = repr_key

    def write(self, adata: AnnData, X: Union[np.ndarray, csr_matrix]) -> None:
        """
        Inserts a new AnnData representation.
        """
        adm.set_value(
            adata=adata, key=self.repr_key, field="obsm", value=X, persist="pipeline"
        )

    def get(self, adata: AnnData) -> np.ndarray:
        """
        Returns a matrix view of a given AnnData.
        """
        X = adm.get_value(adata=adata, key=self.repr_key)
        assert X is not None, "Representation has not been computed."
        if isinstance(X, csr_matrix):
            X = X.toarray()
        return X


class HasMetadata(ABC):
    """
    This trait allows a module to retrieve and store metadata
    from an AnnData object.
    """

    def __init__(self):
        self.metadata: List[Dict[Hashable, Any]] = []

    def retrieve_all_metadata(self, datasets: List[AnnData]) -> None:
        """
        Retrieves necessary information from a list of AnnData objects.
        """
        for adata in datasets:
            self.metadata.append(self.retrieve_metadatata(adata))

    @abstractmethod
    def retrieve_metadatata(self, adata: AnnData) -> Dict[Hashable, Any]:
        """
        This must be implemented by child classes.
        """
        pass

    def get_metadata(self, index: int, key: Hashable) -> Optional[Any]:
        """
        Returns a stored information, or None if it has not be
        registered.
        """
        if index >= len(self.metadata):
            logger.warn(
                f"List index {index} out of range for list "
                f"of size {len(self.metadata)}."
            )
            return None
        return self.metadata[index].get(key, None)


class UsesReference:
    """
    This trait is shared by objects that must use a reference
    dataset.
    """

    def __init__(self):
        pass

    def get_reference_index(self, datasets: List[AnnData]) -> int:
        """
        Returns index of AnnData that has been chosen as a reference. If
        found none or several, returns -1.
        """
        ref_id = -1
        for k, adata in enumerate(datasets):
            is_ref = adm.get_value(adata, AnnDataKeyIdentifiers.IsReference)
            if is_ref is not None:
                if ref_id != -1:
                    return -1  # Several found
                ref_id = k
        return ref_id

    def get_reference(self, datasets: List[AnnData]) -> Optional[AnnData]:
        """
        Returns AnnData that has been chosen as a reference. If
        found none or several, returns None.
        """
        ref_id = self.get_reference_index(datasets)
        if ref_id == -1:
            return None
        return datasets[ref_id]
