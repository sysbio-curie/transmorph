#!/usr/bin/env python3

from anndata import AnnData
from typing import Any, Dict, List

_TypeMetadata = Dict[str, Any]


class HasMetadata:
    """
    This trait allows a module to retrieve and store metadata
    from an AnnData object, in a safe manner. In transmorphh, metadata
    has to be understood as "general information about a dataset" and
    must not be confused with for instance sample labels, which have
    its own trait. This trait is useful to retrieve information in
    AnnData objects for classes that are not allowed to handle these
    objects in their processing method.

    Metadata is handled in a way as safe as possible. When a class
    inherits from HasMetadata, it must explicitly provide all metadata
    keys that will be used, and provide a default value for each of
    those in case it is missing in the AnnData object. Any metadata
    key that would be set without having been declared will raise
    a runtime error.

    All classes inheriting from HasMetadata are expected to implement
    their own retrieve_metadata method, which returns gathered metadata
    as a dictionary.

    Parameters
    ----------
    default_values: Dict[str, Any]
        Dictionary used to declare metadata keys and set their default
        values. Declared metadata keys are exactly keys of this
        dictionary, and their default values are the corresponding
        values in the dictionary.

    Attributes
    ----------
    metadata: List[Dict[str, Any]]
        For each dataset passed to retrieve_all_metadata, its metadata
        is stored in this list of dictionaries at the corresponding
        index.
    """

    def __init__(self, default_values: _TypeMetadata):
        self.metadata_default = default_values.copy()
        self.metadata: List[_TypeMetadata] = []

    def retrieve_all_metadata(self, datasets: List[AnnData]) -> None:
        """
        Retrieves necessary information from a list of AnnData objects.
        Completes with default values provided. This method must NOT
        be overwrittent by child classes, see retrieve_metadata instead.

        Parameters
        ----------
        datasets: List[AnnData]
            AnnData objects to retrieve metadata from.
        """
        for adata in datasets:
            metadata_qry = self.retrieve_metadatata(adata)
            metadata_to_save = self.metadata_default.copy()
            for key in metadata_qry:
                if key not in metadata_to_save:
                    raise ValueError(
                        f"Unexpected key {key}. Allowed keys are "
                        f"{','.join(self.metadata_default.keys())}."
                    )
                metadata_to_save[key] = metadata_qry[key]
            self.metadata.append(metadata_to_save)

    def retrieve_metadatata(self, adata: AnnData) -> _TypeMetadata:
        """
        Given an AnnData objects, extracts relevant metadata and
        store them in a dictionary with the appropriate acces keys.
        These metadata can then be accessed in any other method
        using get_metadata(index, key). This method must be overwritten
        by child classes.

        Parameters
        ----------
        adata: AnnData
            AnnData object to retrieve anndata from.
        """
        raise NotImplementedError

    def get_metadata(self, index: int, key: str) -> Any:
        """
        Returns stored metadata at a given index and key using
        cached information.

        Parameters
        ----------
        index: int
            Dataset index, must coincidate with the one passed in
            retrieve_all_metadata.

        key: str
            Metadata key to retrieve, must have been declared in
            the initialization.
        """
        assert key in self.metadata_default
        print(self.metadata)
        return self.metadata[index][key]
