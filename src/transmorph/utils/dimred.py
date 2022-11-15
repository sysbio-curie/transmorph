#!/usr/bin/env python3

from logging import warn
from typing import Optional
import numpy as np

from sica.base import StabilizedICA
from sklearn.decomposition import PCA
from umap.umap_ import UMAP, find_ab_params

from .matrix import contains_duplicates, perturbate, scale_matrix
from .._logging import logger


def pca_projector(
    X: np.ndarray,
    n_components: int = 2,
    use_subset: Optional[np.ndarray] = None,
) -> PCA:
    """
    Returns a fitted sklearn PCA object.

    Parameters
    ----------
    X: np.ndarray
        (n, d) vectorized dataset to apply PCA on.

    n_components: int, default = 2
        Number of principal components to use. If > d, then it is set to d.

    use_subset: Optional[np.ndarray]
        Optional slice selector (n,) to pass. If passed, the specified subset
        of X is used to fit PCA.
    """
    from .._settings import settings

    assert n_components > 0, "Number of components must be positive."
    n, d = X.shape
    if n_components > d:
        warn(
            "Number of components higher than dataset dimensionality."
            f"Setting it to {d} instead."
        )
        n_components = d

    if use_subset is None:
        X_fit = X
    else:
        X_fit = X[use_subset]

    pca = PCA(n_components=n_components, random_state=settings.global_random_seed)
    pca.fit(X_fit)
    return pca


def pca(
    X: np.ndarray,
    n_components: int = 2,
    use_subset: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Small wrapper for sklearn.decomposition.PCA

    Parameters
    ----------
    X: np.ndarray
        (n, d) vectorized dataset to apply PCA on.

    n_components: int, default = 2
        Number of principal components to use. If > d, then it is set to d.

    use_subset: Optional[np.ndarray]
        Optional slice selector (n,) to pass. If passed, the specified subset
        of X is used to fit PCA.
    """
    pca = pca_projector(X, n_components, use_subset)
    return pca.transform(X)


def ica(
    X: np.ndarray,
    n_components: int = 30,
    max_iter: int = 1000,
    n_runs: int = 10,
    normalize_features: bool = True,
) -> np.ndarray:
    """
    Computes an ICA representation of the data using StablilizedICA.
    """
    sica = StabilizedICA(
        n_components=n_components,
        max_iter=max_iter,
        n_runs=n_runs,
        n_jobs=-1,
    )
    if normalize_features:
        X = scale_matrix(X, axis=0, std_mode=False)

    return sica.fit_transform(X)


def umap(X: np.ndarray, embedding_dimension: int = 2) -> np.ndarray:
    """
    Computes a UMAP representation of dataset X.
    """
    from .._settings import settings

    nsamples = X.shape[0]
    n_epochs = (
        settings.umap_maxiter
        if settings.umap_maxiter is not None
        else 500
        if nsamples < settings.large_dataset_threshold
        else 200
    )
    if settings.umap_a is None or settings.umap_b is None:
        a, b = find_ab_params(settings.umap_spread, settings.umap_min_dist)
    else:
        a, b = settings.umap_a, settings.umap_b

    X_red = pca(X, n_components=settings.neighbors_n_pcs)
    if contains_duplicates(X_red):
        logger.debug("umap > Duplicates detected. Jittering data.")
        X_red = perturbate(X_red, std=0.04)

    return UMAP(
        n_neighbors=settings.umap_n_neighbors,
        n_components=embedding_dimension,
        metric=settings.umap_metric,
        metric_kwds=settings.umap_metric_kwargs,
        n_epochs=n_epochs,
        a=a,
        b=b,
        random_state=settings.umap_random_state,
        negative_sample_rate=settings.umap_negative_sample_rate,
        learning_rate=settings.umap_alpha,
    ).fit_transform(X_red)
