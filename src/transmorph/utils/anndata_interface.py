#!/usr/bin/env python3

from anndata import AnnData
from typing import List

import numpy as np

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
    if "transmorph" not in adata.uns:
        return False
    if "infos" not in adata.uns["transmorph"]:
        return False
    return dataset_key in adata.uns["transmorph"]["infos"]


def common_genes(datasets: List[AnnData]) -> np.ndarray:
    """
    Returns common genes between datasets.
    """
    if len(datasets) == 0:
        return np.array([])
    if len(datasets) == 1:
        return datasets[0].var_names.to_numpy()
    common_genes = datasets[0].var_names
    for adata in datasets[1:]:
        common_genes = common_genes.intersection(adata.var_names)
    return common_genes.to_numpy()
