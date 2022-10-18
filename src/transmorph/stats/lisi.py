#!/usr/bin/env python3

import numba
import numpy as np

from anndata import AnnData
from typing import List, Dict, Optional, Union

from ..utils.dimred import pca
from ..utils.matrix import sort_sparse_matrix, extract_chunks
from ..utils.graph import nearest_neighbors_custom


def lisi(
    datasets: Union[List[AnnData], Dict[str, AnnData]],
    obsm: Optional[str] = None,
    obs: Optional[str] = None,
    perplexity: int = 15,
    n_pcs: int = 15,
) -> List[np.ndarray]:
    """
    LISI statistic measures how heterogeneous a sample neighborhood
    is for a certain label. Is is notably used in the Harmony
    integration pipeline to measure how well integrated datasets
    are.

    Parameters
    ----------
    datasets: Union[List[AnnData], Dict[str, AnnData]]
        Set of batches.

    obsm: Optional[str]
        Representation matrix to use, by default uses AnnData.X.

    obs: Optional[str]
        Categorical observation to use. If none is provided, batches
        are used as categories.

    n_pcs: int = 15
        Number of principal components to use if matrix representation
        is of greater dimension.

    Returns
    -------
    For every dataset, its point-wise LISI score.
    """
    if isinstance(datasets, Dict):
        datasets = list(datasets.values())

    if obsm is None:
        X = np.concatenate([adata.X for adata in datasets], axis=0)
    else:
        X = np.concatenate([adata.obsm[obsm] for adata in datasets], axis=0)

    if X.shape[1] > n_pcs:
        X = pca(X, n_components=n_pcs)

    if obs is None:
        labels = np.array(
            sum([[i] * adata.n_obs for i, adata in enumerate(datasets)], [])
        )
    else:
        labels_raw = np.concatenate([adata.obs[obs] for adata in datasets])
        labels = np.zeros(labels_raw.shape, dtype=int)
        for i, lb in enumerate(set(labels_raw)):
            labels[labels_raw == lb] = i

    return extract_chunks(
        compute_lisi(X, labels, perplexity),
        [adata.n_obs for adata in datasets],
    )


# Adapted from harmonypy:
# https://github.com/slowkow/harmonypy/blob/master/harmonypy/lisi.py
def compute_lisi(
    X: np.ndarray,
    labels: np.ndarray,
    perplexity: float = 15.0,
) -> np.ndarray:
    """
    LISI statistic measures how heterogeneous a sample neighborhood
    is for a certain label. Is is notably used in the Harmony
    integration pipeline to measure how well integrated datasets
    are.

    Parameters
    ----------
    X: np.ndarray
        (N, d) Concatenated data matrices in the embedding space

    labels: np.ndarray
        (N,) Labels of the observations

    perplexity: float, default = 30.0
        Neighborhood size.
    """

    # n_neighbors >= 3*perplexity
    connectivity = nearest_neighbors_custom(
        X, mode="edges", n_neighbors=int(perplexity * 3)
    )
    indices, _ = sort_sparse_matrix(connectivity)
    label_per_nb = labels[indices]

    all_labels = np.unique(labels)
    simpson = compute_simpson(label_per_nb, all_labels)

    return 1.0 / simpson


@numba.njit(fastmath=True)
def compute_simpson(label_per_nb: np.ndarray, labels_set: np.ndarray) -> np.ndarray:
    """
    Helper function for compute_lisi, returns simpson index for
    each sample, for one label
    """
    nlabels = labels_set.shape[0]
    nsamples, n_neighbors = label_per_nb.shape
    label_frequencies = np.zeros((nsamples, nlabels), dtype=np.float32)

    for i in range(nsamples):
        labels_i = label_per_nb[i]
        for lj, l in enumerate(labels_set):
            label_frequencies[i, lj] = (labels_i == l).sum()

    label_frequencies /= n_neighbors
    label_frequencies *= label_frequencies
    return np.sum(label_frequencies, axis=1)
