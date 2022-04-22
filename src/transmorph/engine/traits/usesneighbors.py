#!/usr/bin/env python3

import warnings

from anndata import AnnData
from scipy.sparse import csr_matrix
from typing import Dict, List, Literal

from ...utils import (
    anndata_manager as adm,
    AnnDataKeyIdentifiers,
    nearest_neighbors,
)


class UsesNeighbors:
    """
    Gives the ability to manipulate nearest neighbors.
    """

    # Will be turned to true if any class with this trait is initialized.
    Used = False
    # Cache of neighbor graphs
    NeighborsGraphs = []

    def __init__(self):
        UsesNeighbors.Used = True

    @staticmethod
    def set_settings_to_scanpy(adata: AnnData) -> None:
        """
        Retrieves scanpy neighbors parameters, and set transmorph settings
        accordingly. If parameters are not found, transmorph parameters are
        left untouched.
        """
        from ..._settings import settings

        parameters = adm.get_value(
            adata=adata,
            key="neighbors",
            transmorph_key=False,
            field="uns",
        )
        if not isinstance(parameters, Dict):
            return
        parameters = parameters.get("params", None)
        if parameters is None:
            warnings.warn("No information found for scanpy neighbors.")
            settings.neighbors_use_scanpy = False
            return
        # We want to raise an error if n_neighbors or metric is missing,
        # as this is unexpected.
        settings.n_neighbors = parameters["n_neighbors"]
        settings.neighbors_metric = parameters["metric"]
        if "metric_kwds" in parameters:
            settings.neighbors_metric_kwargs = parameters["metric_kwds"]
        if "n_pcs" in parameters:
            settings.neighbors_n_pcs = parameters["n_pcs"]

    @staticmethod
    def compute_neighbors_graphs(
        datasets: List[AnnData],
        representation_key: AnnDataKeyIdentifiers,
    ) -> None:
        """
        Computes a neighbors graph, and stores it in adata.
        """
        from ..._settings import settings

        use_scanpy = settings.neighbors_use_scanpy
        for adata in datasets:
            if not use_scanpy:
                X = adm.get_value(adata, representation_key)
                # Settings parameters are used as default
                matrix = nearest_neighbors(
                    X=X,
                    include_self_loops=False,
                    symmetrize=False,
                    mode="distances",
                )
            else:  # Scanpy mode
                matrix = adm.get_value(
                    adata=adata,
                    key="distances",
                    transmorph_key=False,
                    field="obsp",
                )
                if matrix is None:
                    raise ValueError("Scanpy neighbors not found.")
            assert isinstance(matrix, csr_matrix)
            UsesNeighbors.NeighborsGraphs.append(matrix)

    @staticmethod
    def get_neighbors_graph(
        idx: int,
        mode: Literal["edges", "distances"] = "edges",
    ) -> csr_matrix:
        """
        Returns nearest neighbors data for dataset #idx.

        Parameters
        ----------
        idx: int
            Dataset indice when called compute_neighbor_graphs.
        """
        assert mode in ("edges", "distances")
        nn_matrix = UsesNeighbors.NeighborsGraphs[idx]
        if mode == "edges":
            nn_matrix = nn_matrix.astype(bool)
        return nn_matrix
