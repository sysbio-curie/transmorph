#!/usr/bin/env python3

from anndata import AnnData

import numpy as np


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
    assert not isset_matrix(adata, dataset_key)
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
    assert isset_matrix(adata, dataset_key)
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
    if "transmorph" not in adata.uns:
        return False
    if "matrices" not in adata.uns["transmorph"]:
        return False
    return dataset_key in adata.uns["transmorph"]["matrices"]


def set_attribute(adata: AnnData, dataset_key: str, X) -> None:
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
    assert not isset_attribute(adata, dataset_key)
    if "transmorph" not in adata.uns:
        adata.uns["transmorph"] = {}
    if "attributes" not in adata.uns["transmorph"]:
        adata.uns["transmorph"]["attributes"] = {}
    adata.uns["transmorph"]["attributes"][dataset_key] = X


def get_attribute(adata: AnnData, dataset_key: str):
    """
    Retrieves an object stored in the AnnData object by set_attribute.

    Parameters
    ----------
    adata: AnnData
        Target dataset

    dataset_key: str
        Target attribute identifier

    Returns
    -------
    The required np.ndarray.
    """
    assert isset_attribute(adata, dataset_key)
    return adata.uns["transmorph"]["attributes"][dataset_key]


def delete_attribute(adata: AnnData, dataset_key: str) -> None:
    """
    Deletes the attribute stored in the AnnData object by set_attribute.

    Parameters
    ----------
    adata: AnnData
        Target dataset

    dataset_key: str
        Target attribute identifier
    """
    assert isset_attribute(adata, dataset_key)
    del adata.uns["transmorph"]["attributes"][dataset_key]


def isset_attribute(adata: AnnData, dataset_key: str) -> bool:
    """
    Detects if an attribute is registered under $dataset_key.

    Parameters
    ----------
    adata: AnnData
        Target dataset

    dataset_key: str
        Target attribute identifier
    """
    if "transmorph" not in adata.uns:
        return False
    if "attributes" not in adata.uns["transmorph"]:
        return False
    return dataset_key in adata.uns["transmorph"]["attributes"]
