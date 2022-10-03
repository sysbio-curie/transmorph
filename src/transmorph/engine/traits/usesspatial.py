#!/usr/bin/env python3

import numpy as np

from anndata import AnnData
from typing import List


class UsesSpatial:
    """
    TODO
    """

    def __init__(self):
        self._spatial_coordinates_cache: List[np.ndarray] = []

    def retrieve_spatial_coordinates(self, datasets: List[AnnData]) -> None:
        self._spatial_coordinates_cache = []
        for adata in datasets:
            if "spatial" not in adata.obsm:
                raise ValueError("No .obsm['spatial'] found in AnnData.")
            self._spatial_coordinates_cache.append(adata.obsm["spatial"])

    def get_spatial_coordinates(self, index: int) -> np.ndarray:
        return self._spatial_coordinates_cache[index].copy()
