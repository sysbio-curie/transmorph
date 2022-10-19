#!/usr/bin/env python3

import numpy as np

import anndata
import random
import string
from typing import Any, Union, Tuple, Type

MAX_ITER = 500


def assert_type(value: Any, allowed: Union[Type, Tuple[Type, ...]]) -> None:
    """
    Small helper to type check.
    """
    if isinstance(value, allowed):
        return
    if isinstance(allowed, Type):
        str_allowed = allowed.__name__
    else:
        str_allowed = ", ".join([_type.__name__ for _type in allowed])
    raise TypeError(f"Unexpected type: {type(value)}. Expected {str_allowed}.")


def rand_str(ln: int) -> str:
    # Just a random string generator
    return "".join(random.choices(string.ascii_letters, k=ln))


def generate_str_elements(
    n: int = 100,
    ln: int = 8,
    conserve_duplicates: bool = False,
) -> np.ndarray:
    # Helper function generating random features
    elements = np.array([rand_str(ln) for _ in range(n)])
    if conserve_duplicates:
        return elements
    return np.unique(elements)


def generate_anndata(
    obs: Union[int, np.ndarray] = 100,
    var: Union[int, np.ndarray] = 20,
    target_sparsity: float = 0.8,
):
    # Helper function generating random anndatas.
    if isinstance(obs, int):
        obs = np.unique(generate_str_elements(obs, 6))
    if isinstance(var, int):
        var = np.unique(generate_str_elements(var, 6))
    nobs, nvar = obs.shape[0], var.shape[0]
    X = np.random.random(size=(nobs, nvar))
    X = X * (np.random.random(size=X.shape) > target_sparsity)
    adata = anndata.AnnData(X, dtype=X.dtype)
    adata.obs[None] = obs
    adata.obs = adata.obs.set_index(None)
    adata.var[None] = var
    adata.var = adata.var.set_index(None)
    return adata
