#!/usr/bin/env python3

from logging import warn
from typing import List, Tuple, Union
import numpy as np

from sklearn.decomposition import PCA


def pca(
    X: np.ndarray, n_components: int = 2, return_transformation: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Small wrapper for sklearn.decomposition.PCA

    Parameters
    ----------
    X: np.ndarray of shape (n, d)
        Vectorized dataset to apply PCA on.

    n_components: int, default = 2
        Number of principal components to use. If > d, then it is set to d.

    return_transformation: bool, default = False
        Return the components matrix together with the transformed dataset.
    """
    assert n_components > 0, "Number of components must be positive."
    n, d = X.shape
    if n_components > d:
        warn(
            "Number of components higher than dataset dimensionality."
            f"Setting it to {d} instead."
        )
        n_components = d
    pca = PCA(n_components=n_components)
    pca.fit(X)
    if return_transformation:
        return pca.transform(X), pca.components_
    return pca.transform(X)


def pca_multi(
    Xs: List[np.ndarray],
    n_components: int = 2,
    strategy: str = "concatenate",
    return_transformation: bool = False,
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], np.ndarray]]:
    """
    Embeds a set of datasets in a common PC space, following one of the following
    strategies:

        - "concatenate": concatenate all datasets together on the axis 0, then
        perform a PCA on this result. Needs all datasets to be in the same
        features space.

        - "reference": project everything on the first dataset PC space. Needs
        all datasets to be in the same features space.

        - "composite": use an average of the transformation matrices to define
        projection PC space. Needs all datasets to be in the same features space.

        - "independent": assume variance axes are preserved between datasets, and
        perform an independent PC projection of same dimensionality for each dataset.

    Parameters
    ----------
    Xs: List[np.ndarray]
        List of vectorized datasets to embed

    n_components: int, default = 2
        Number of PCs to use.

    strategy: str, default = 'concatenate'
        Strategy to choose projection space in 'concatenate', 'reference',
        'composite' and 'independent'

    return_transformation: bool = False
        If strategy != 'independent', returns the transform matrix.
    """
    assert len(Xs) > 0, "No datasets provided."
    if len(Xs) == 1:
        return pca(
            Xs[0],
            n_components=n_components,
            return_transformation=return_transformation,
        )
    assert n_components > 0, "Number of components must be positive."
    d = Xs[0].shape[1]
    if strategy != "independent":
        assert all(
            X.shape[1] == d for X in Xs
        ), "All datasets must be of similar dimensionality."
    if strategy == "concatenate":
        pc_result = pca(
            np.concatenate(Xs, axis=0),
            n_components=n_components,
            return_transformation=return_transformation,
        )
        Xs_pca, components = None, None
        if return_transformation:
            Xs_pca, components = pc_result
        else:
            Xs_pca = pc_result
        offset, datasets = 0, []
        for X in Xs:
            datasets.append(Xs_pca[offset : offset + X.shape[0]])
            offset += X.shape[0]
        if return_transformation:
            return (datasets, components)
        return datasets
    elif strategy == "reference":
        X_ref_pca, components = pca(
            Xs[0], n_components=n_components, return_transformation=True
        )
        datasets = [X_ref_pca]
        for X in Xs[1:]:
            datasets.append(X @ components)
        if return_transformation:
            return (datasets, components)
        return datasets
    elif strategy == "composite":
        total_pca = np.zeros((Xs[0].shape[1], n_components))
        for X in Xs:
            _, transform = pca(X, n_components=n_components, return_transformation=True)
            total_pca += transform
        total_pca /= len(Xs)
        datasets = [X @ total_pca for X in Xs]
        if return_transformation:
            return datasets, total_pca
        return datasets
    elif strategy == "independent":
        assert (
            return_transformation is False
        ), "No common transformation can be returned if strategy == 'independent'"
        return [pca(X, n_components=n_components) for X in Xs]
    else:
        raise NotImplementedError(
            "Strategy must be in 'concatenate', 'reference', 'composite'."
        )
