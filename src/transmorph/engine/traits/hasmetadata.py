#!/usr/bin/env python3

from anndata import AnnData
from typing import Any, Dict, Hashable, List, Optional


class HasMetadata:
    """
    This trait allows a module to retrieve and store metadata
    from an AnnData object.
    TODO handle subsampling when metadata is related to samples
    """

    def __init__(self):
        self.metadata: List[Dict[Hashable, Any]] = []

    def retrieve_all_metadata(self, datasets: List[AnnData]) -> None:
        """
        Retrieves necessary information from a list of AnnData objects.
        """
        for adata in datasets:
            self.metadata.append(self.retrieve_metadatata(adata))

    def retrieve_metadatata(self, adata: AnnData) -> Dict[Hashable, Any]:
        """
        This must be implemented by child classes.
        """
        raise NotImplementedError

    def get_metadata(self, index: int, key: Hashable) -> Optional[Any]:
        """
        Returns a stored information, or None if it has not be
        registered.
        """
        if index >= len(self.metadata):
            return None
        return self.metadata[index].get(key, None)
