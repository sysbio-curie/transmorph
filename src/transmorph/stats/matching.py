#!/usr/bin/env python3

import numpy as np

from anndata import AnnData
from scipy.sparse import csr_matrix
from typing import Optional


def base_edge_quality(
    adata1: AnnData,
    adata2: AnnData,
    matching: csr_matrix,
    label: str,
    tp_cost: float,
    fp_cost: float,
    custom_nsamples: Optional[int] = None,
) -> float:
    """
    Computes a quality metric for a matching between two datasets whose
    point labels are known. For every point $x matched with $y1 .. $yn,
    its score is given by

    sum_{i=1..n} M(x, yi)*eq(x, yi) / sum_{i=1..n} M(x, yi)

    where M(x, yi) is the matching strength between x and yi, and
    eq(x, yi) = tp_cost if label(x) == label(yi), fp_cost otherwise.

    Then, edge accuracy is given as the average over all points of this
    score. A zero edge accuracy is interpreted as a matching decorrelated
    with labels, negative as an anticorrelated matching/labels and positive
    means matches are positively correlated with labels.

    A bad edge accuracy is associated with a high chance of poor integration
    quality, as matching edges are the skeleton of the final embedding.

    Parameters
    ----------
    adata1: AnnData
        First dataset (in rows in matching), n_obs = n1

    adata2: AnnData
        Second dataset (in columns in matching), n_obs = n2

    matching: csr_matrix
        Matching edges represented as a matrix of shape (n1, n2)

    label: str
        adata.obs key for labels to compare. Must be the same for
        adata1 and adata2.

    tp_cost: float
        Cost associated to an edge matching two samples of the same class.

    fp_cost: float
        Cost associated to an edge matching two samples of different classes.

    custom_nsamples: int, Optional
        By default, normalizes the cost by total number of samples. In case
        data has been subsampled, provide the subsample size using this parameter.
    """
    # Sanity checks
    assert label in adata1.obs
    assert label in adata2.obs

    n1, n2 = adata1.n_obs, adata2.n_obs
    assert matching.shape == (n1, n2)

    label1 = adata1.obs[label].to_numpy()
    label2 = adata2.obs[label].to_numpy()
    counts1 = np.array(matching.sum(axis=1))[:, 0]
    counts2 = np.array(matching.sum(axis=0))[0]
    accuracy = np.zeros(n1 + n2)
    matching = matching.tocoo().astype(float)
    for i1, i2, v in zip(matching.row, matching.col, matching.data):
        if label1[i1] != label2[i2]:
            v = fp_cost * v
        else:
            v = tp_cost * v
        # Normalizing by row/col marginals
        accuracy[i1] += v / counts1[i1]
        accuracy[n1 + i2] += v / counts2[i2]
    if custom_nsamples is None:
        # By default, use all samples
        custom_nsamples = accuracy.shape[0]
    return accuracy.sum() / custom_nsamples


def edge_accuracy(
    adata1: AnnData,
    adata2: AnnData,
    matching: csr_matrix,
    label: str,
    custom_nsamples: Optional[int] = None,
) -> float:
    """
    Computes an accuracy metric for a matching between two datasets whose
    point labels are known. For every point $x matched with $y1 .. $yn,
    its score is given by

    sum_{i=1..n} M(x, yi)*eq(x, yi) / sum_{i=1..n} M(x, yi)

    where M(x, yi) is the matching strength between x and yi, and
    eq(x, yi) = 1 if label(x) == label(yi), -1 otherwise.

    Then, edge accuracy is given as the average over all points of this
    score. A zero edge accuracy is interpreted as a matching decorrelated
    with labels, negative as an anticorrelated matching/labels and positive
    means matches are positively correlated with labels.

    A bad edge accuracy is associated with a high chance of poor integration
    quality, as matching edges are the skeleton of the final embedding.

    Parameters
    ----------
    adata1: AnnData
        First dataset (in rows in matching), n_obs = n1

    adata2: AnnData
        Second dataset (in columns in matching), n_obs = n2

    matching: csr_matrix
        Matching edges represented as a matrix of shape (n1, n2)

    label: str
        adata.obs key for labels to compare. Must be the same for
        adata1 and adata2.

    custom_nsamples: int, Optional
        By default, normalizes the cost by total number of samples. In case
        data has been subsampled, provide the subsample size using this parameter.
    """
    return base_edge_quality(adata1, adata2, matching, label, 1, 0, custom_nsamples)