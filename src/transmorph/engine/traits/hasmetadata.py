#!/usr/bin/env python3

from anndata import AnnData
from typing import Any, Dict, List, Optional


class HasMetadata:
    """
    This trait allows a module to retrieve and store metadata
    from an AnnData object, in a safe manner.
    TODO handle subsampling when metadata is related to samples

    Parameters
    ----------

    """

    def __init__(self, default_values: Dict[str, Any]):
        self.metadata_default = default_values.copy()
        self.metadata: List[Dict[str, Any]] = []

    def retrieve_all_metadata(self, datasets: List[AnnData]) -> None:
        """
        Retrieves necessary information from a list of AnnData objects.
        Completes with default values provided.
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

    def retrieve_metadatata(self, adata: AnnData) -> Dict[str, Any]:
        """
        This must be implemented by child classes.
        """
        raise NotImplementedError

    def get_metadata(self, index: int, key: str) -> Optional[Any]:
        """
        Returns a stored information, or None if it has not be
        registered.
        """
        assert key in self.metadata_default
        if index >= len(self.metadata):
            return self.metadata_default[key]
        return self.metadata[index][key]
