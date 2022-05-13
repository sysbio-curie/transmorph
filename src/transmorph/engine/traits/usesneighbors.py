#!/usr/bin/env python3

import numpy as np

from anndata import AnnData
from scipy.sparse import csr_matrix
from typing import List, Literal, Optional, Tuple, Union

from ...utils.anndata_manager import (
    anndata_manager as adm,
    AnnDataKeyIdentifiers,
)
from ...utils.graph import nearest_neighbors
from ...utils.matrix import sort_sparse_matrix, sparse_from_arrays

_ModeNeighborsMatrix = Literal["csr", "arrays"]
_TypeNeighborsMatrix = Union[csr_matrix, Tuple[np.ndarray, Optional[np.ndarray]]]


class UsesNeighbors:
    """
    Gives the ability to manipulate nearest neighbors.
    Global status is quite dirty, we should find a better way.
    It will cause us trouble in the future.
    """

    # Will be turned to true if any class with this trait is initialized.
    Used = False
    # Cache of neighbor graphs (distance matrix/index matrix)
    NeighborsDistances: List[np.ndarray] = []
    NeighborsIndices: List[np.ndarray] = []

    def __init__(self):
        UsesNeighbors.Used = True

    @staticmethod
    def reset():
        from ... import settings

        UsesNeighbors.Used = False
        UsesNeighbors.NeighborsDistances = []
        UsesNeighbors.NeighborsIndices = []
        settings.n_neighbors_max = settings._n_neighbors_max_init

    @staticmethod
    def compute_neighbors_graphs(
        datasets: List[AnnData],
        representation_key: Optional[AnnDataKeyIdentifiers] = None,
    ) -> None:
        """
        Computes a neighbors graph, and stores it in adata.
        """
        from ..._settings import settings

        UsesNeighbors.reset()

        settings.n_neighbors_max = min(
            settings.n_neighbors_max,
            min(adata.n_obs for adata in datasets),  # Change to max?
        )

        if representation_key is None:
            representation_key = AnnDataKeyIdentifiers.BaseRepresentation

        algorithm = "sklearn"
        if any(adata.n_obs > settings.large_dataset_threshold for adata in datasets):
            algorithm = "nndescent"
        for adata in datasets:
            X = adm.get_value(adata, representation_key)
            matrix = nearest_neighbors(
                X=X,
                mode="distances",
                algorithm=algorithm,
            )
            indices, distances = sort_sparse_matrix(matrix)
            UsesNeighbors.NeighborsIndices.append(indices)
            UsesNeighbors.NeighborsDistances.append(distances)

    @staticmethod
    def get_neighbors_graph(
        idx: int,
        mode: Literal["edges", "distances"] = "edges",
        n_neighbors: Optional[int] = None,
        return_format: _ModeNeighborsMatrix = "csr",
    ) -> _TypeNeighborsMatrix:
        """
        Returns nearest neighbors data for dataset #idx.

        Parameters
        ----------
        idx: int
            Dataset indice when called compute_neighbor_graphs.
        """
        from ... import settings, use_setting

        assert mode in ("edges", "distances")
        assert len(UsesNeighbors.NeighborsDistances) > 0, (
            "UsesNeighbors must be initialized via"
            " UsesNeighbors.compute_neighbors_graphs."
        )
        n_neighbors = use_setting(n_neighbors, settings.n_neighbors_max)
        n_neighbors = min(n_neighbors, settings.n_neighbors_max)
        assert n_neighbors <= settings.n_neighbors_max, (
            f"n_neighbors < n_neighbors_max ({n_neighbors} < "
            f"{settings.n_neighbors_max}). You can increase this "
            "value by increasing transmorph.settings.n_neighbors_max."
        )
        nn_indices = UsesNeighbors.NeighborsIndices[idx]
        nn_indices = nn_indices[:, :n_neighbors]
        if mode == "distances":
            nn_distances = UsesNeighbors.NeighborsDistances[idx]
            nn_distances = nn_distances[:, :n_neighbors]
            if return_format == "csr":
                return sparse_from_arrays(nn_indices, nn_distances)
            return nn_indices, nn_distances
        if return_format == "csr":
            return sparse_from_arrays(nn_indices).astype(float)
        return nn_indices, None
