#!/usr/bin/env python3

from anndata import AnnData
from typing import List, Optional, TypeVar
from ...utils.anndata_manager import anndata_manager as adm, AnnDataKeyIdentifiers

T = TypeVar("T")


class UsesReference:
    """
    This trait is shared by objects that must use a reference
    dataset.
    """

    def __init__(self):
        self.reference_index: Optional[int] = None

    @staticmethod
    def write_is_reference(adata: AnnData) -> None:
        """
        Sets an anndata to be considered as reference. Should not be used on
        more than one anndata object.
        """
        adm.set_value(
            adata=adata,
            key=AnnDataKeyIdentifiers.IsReference,
            field="uns",
            value=True,
            persist="pipeline",
        )

    def retrieve_reference_index(self, datasets: List[AnnData]) -> None:
        """
        Returns index of AnnData that has been chosen as a reference. If
        found none or several, returns -1.
        """
        ref_id = -1
        for k, adata in enumerate(datasets):
            is_ref = adm.get_value(adata, AnnDataKeyIdentifiers.IsReference)
            if is_ref is not None:
                if ref_id != -1:
                    raise AttributeError("More than one reference.")
                ref_id = k
        if ref_id == -1:
            return
        self.reference_index = ref_id

    def get_reference_item(self, datasets: List[T]) -> Optional[T]:
        """
        Returns object that has been chosen as a reference from a list. If
        found none or several, returns None.
        """
        ref_id = self.reference_index
        if ref_id is None:
            return None
        return datasets[ref_id]
