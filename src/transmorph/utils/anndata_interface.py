#!/usr/bin/env python3

from anndata import AnnData
from typing import List, Optional

import numpy as np
import scanpy as sc
import warnings

# This file implements a standardized way for Transmorph modules
# to interact with annotated data objects. Two data types can
# be manipulated:
#
# - matrix, to store alternative representations of datasets
# - info, to store scalar or str values (e.g. pipeline parameters,
#   dataset metric...)


def set_matrix(adata: AnnData, dataset_key: str, X: np.ndarray) -> None:
    """
    Registers a matrix in an AnnData object, under a unique string identifier.

    Parameters
    ----------
    adata: AnnData
        Target dataset

    dataset_key: str
        Target matrix identifier

    X: np.ndarray
        Matrix to write
    """
    warnings.warn("Using anndata interface is deprecated.")
    if "transmorph" not in adata.uns:
        adata.uns["transmorph"] = {}
    if "matrices" not in adata.uns["transmorph"]:
        adata.uns["transmorph"]["matrices"] = {}
    adata.uns["transmorph"]["matrices"][dataset_key] = X


def get_matrix(adata: AnnData, dataset_key: str) -> np.ndarray:
    """
    Retrieves a matrix stored in the AnnData object by set_matrix.

    Parameters
    ----------
    adata: AnnData
        Target dataset

    dataset_key: str
        Target matrix identifier

    Returns
    -------
    The required np.ndarray.
    """
    warnings.warn("Using anndata interface is deprecated.")
    if dataset_key == "":
        return adata.X
    assert isset_matrix(adata, dataset_key)
    return adata.uns["transmorph"]["matrices"][dataset_key]


def delete_matrix(adata: AnnData, dataset_key: str) -> None:
    """
    Deletes the matrix stored in the AnnData object by set_matrix.

    Parameters
    ----------
    adata: AnnData
        Target dataset

    dataset_key: str
        Target matrix identifier
    """
    warnings.warn("Using anndata interface is deprecated.")
    if not isset_matrix(adata, dataset_key):
        return
    del adata.uns["transmorph"]["matrices"][dataset_key]


def isset_matrix(adata: AnnData, dataset_key: str) -> bool:
    """
    Detects if an matrix is registered under $dataset_key.

    Parameters
    ----------
    adata: AnnData
        Target dataset

    dataset_key: str
        Target matrix identifier
    """
    warnings.warn("Using anndata interface is deprecated.")
    if dataset_key == "":
        return True
    if "transmorph" not in adata.uns:
        return False
    if "matrices" not in adata.uns["transmorph"]:
        return False
    return dataset_key in adata.uns["transmorph"]["matrices"]


def set_info(adata: AnnData, dataset_key: str, X) -> None:
    """
    Registers an object in an AnnData object, under a unique string identifier.

    Parameters
    ----------
    adata: AnnData
        Target dataset

    dataset_key: str
        Target matrix identifier

    X: python object
        Object to write
    """
    warnings.warn("Using anndata interface is deprecated.")
    if "transmorph" not in adata.uns:
        adata.uns["transmorph"] = {}
    if "infos" not in adata.uns["transmorph"]:
        adata.uns["transmorph"]["infos"] = {}
    adata.uns["transmorph"]["infos"][dataset_key] = X


def get_info(adata: AnnData, dataset_key: str):
    """
    Retrieves an object stored in the AnnData object by set_info.

    Parameters
    ----------
    adata: AnnData
        Target dataset

    dataset_key: str
        Target info identifier

    Returns
    -------
    The required np.ndarray.
    """
    warnings.warn("Using anndata interface is deprecated.")
    assert isset_info(adata, dataset_key)
    return adata.uns["transmorph"]["infos"][dataset_key]


def delete_info(adata: AnnData, dataset_key: str) -> None:
    """
    Deletes the info stored in the AnnData object by set_info.

    Parameters
    ----------
    adata: AnnData
        Target dataset

    dataset_key: str
        Target info identifier
    """
    warnings.warn("Using anndata interface is deprecated.")
    if not isset_info(adata, dataset_key):
        return
    del adata.uns["transmorph"]["infos"][dataset_key]


def isset_info(adata: AnnData, dataset_key: str) -> bool:
    """
    Detects if an info is registered under $dataset_key.

    Parameters
    ----------
    adata: AnnData
        Target dataset

    dataset_key: str
        Target info identifier
    """
    warnings.warn("Using anndata interface is deprecated.")
    if "transmorph" not in adata.uns:
        return False
    if "infos" not in adata.uns["transmorph"]:
        return False
    return dataset_key in adata.uns["transmorph"]["infos"]


def highly_variable_genes(adata: AnnData, n_top_genes: int) -> np.ndarray:
    """
    Substitute for highly variable genes from scanpy.
    """
    warnings.warn("Using anndata interface is deprecated.")
    if n_top_genes >= adata.n_vars:
        return adata.var_names.to_numpy()
    # Patching a possible error in scanpy pipeline
    if "log1p" in adata.uns and "base" not in adata.uns["log1p"]:
        adata.uns["log1p"]["base"] = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    return adata.var_names[adata.var.highly_variable].to_numpy()


def common_genes(
    datasets: List[AnnData], n_top_var: Optional[int] = None
) -> np.ndarray:
    """
    Returns common genes between datasets.
    """
    warnings.warn("Using anndata interface is deprecated.")
    if len(datasets) == 0:
        return np.array([])
    adata = datasets[0]
    if n_top_var is None:
        common_genes = datasets[0].var_names.to_numpy()
    else:
        common_genes = highly_variable_genes(datasets[0], n_top_genes=n_top_var)
    if len(datasets) == 1:
        return common_genes
    for adata in datasets[1:]:
        if n_top_var is None:
            adata_genes = adata.var_names.to_numpy()
        else:
            adata_genes = highly_variable_genes(adata, n_top_genes=n_top_var)
        common_genes = np.intersect1d(common_genes, adata_genes)
    return common_genes
