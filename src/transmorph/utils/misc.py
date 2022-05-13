#!/usr/bin/env python3

import numpy as np

import anndata
import random
import string
from typing import Any, Optional, Union, Tuple, Type

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


def generate_features(
    n: int = 100,
    ln: int = 8,
    leave_duplicates: bool = False,
) -> np.ndarray:
    # Helper function generating random features
    features = np.array([rand_str(ln) for _ in range(n)])
    if leave_duplicates:
        return features
    return np.unique(features)


def generate_anndata(
    nobs: int,
    nvars: int,
    features_set: Optional[np.ndarray],
    force_shape: bool = False,
):
    # Helper function generating random anndatas of size approx.
    # nobs x nvars
    if features_set is None:
        features_set = np.unique(generate_features(nobs * 10, 10))
    for _ in range(MAX_ITER):
        features = np.unique(random.choices(features_set, k=nvars))
        X = np.random.random(size=(nobs, features.shape[0]))
        adata = anndata.AnnData(X, dtype=X.dtype)
        adata.var[None] = features
        adata.var = adata.var.set_index(None)
        if not force_shape or adata.X.shape == (nobs, nvars):
            return adata
    raise ValueError("Could not generate an AnnData of the requested shape.")
