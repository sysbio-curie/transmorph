#!/usr/bin/env python3

import numpy as np

from anndata import AnnData
from scipy.sparse import csr_matrix


def edge_accuracy(
    adata1: AnnData, adata2: AnnData, matching: csr_matrix, label: str
) -> float:
    """
    Computes a quality metric for a matching between two datasets whose
    point labels are known. For every point $x matched with $y1 .. $yn,
    its score is given by
    sum_{i=1..n} M(x, yi)*eq(x, yi) / sum_{i=1..n} M(x, yi)
    where M(x, yi) is the matching strength between x and yi, and
    eq(x, yi) = 1 if label(x) == label(yi), -1 otherwise.

    Then, edge accuracy is given as the average over all points of this
    score. A zero edge accuracy is interpreted as a matching decorrelated
    with labels, negative as an anticorrelated matching/labels and positive
    means matches are positively correlated with labels.
    """
    # Sanity checks
    assert label in adata1.obs
    assert label in adata2.obs

    n1, n2 = adata1.n_obs, adata2.n_obs
    assert matching.shape == (n1, n2)

    label1 = adata1.obs.to_numpy()
    label2 = adata2.obs.to_numpy()
    counts1 = np.array(matching.sum(axis=1))[:, 0]
    counts2 = np.array(matching.sum(axis=0))[0]
    accuracy = np.zeros(n1 + n2)
    matching = matching.tocoo().astype(float)
    for i1, i2, v in zip(matching.row, matching.col, matching.data):
        if label1[i1] != label2[i2]:
            v = -v
        accuracy[i1] += v / counts1[i1]
        accuracy[n1 + i2] += v / counts2[i2]
    return accuracy.sum() / accuracy.shape[0]
