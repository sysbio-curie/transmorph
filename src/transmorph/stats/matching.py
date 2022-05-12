#!/usr/bin/env python3

from anndata import AnnData
from scipy.sparse import csr_matrix

from ..utils.matrix import guess_is_discrete
from .._logging import logger


def base_edge_penalty_discrete(
    adata1: AnnData,
    adata2: AnnData,
    matching: csr_matrix,
    label: str,
    tp_cost: float,
    fp_cost: float,
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
    """
    # Sanity checks
    assert label in adata1.obs, f"Label {label} missing in anndata 1."
    assert label in adata2.obs, f"Label {label} missing in anndata 2."

    n1, n2 = adata1.n_obs, adata2.n_obs
    assert matching.shape == (
        n1,
        n2,
    ), f"Unexpected matching shape, {matching.shape} != {(n1, n2)}"

    if matching.count_nonzero() == 0:
        return 0.0

    label1 = adata1.obs[label].to_numpy()
    label2 = adata2.obs[label].to_numpy()

    if not guess_is_discrete(label1):
        logger.warn("Observation 1 does not seem to be discrete. ")
    if not guess_is_discrete(label2):
        logger.warn("Observation 2 does not seem to be discrete. ")

    matching_coo = matching.tocoo().astype(float)
    score = 0.0
    for i1, i2, v in zip(matching_coo.row, matching_coo.col, matching_coo.data):
        if label1[i1] != label2[i2]:
            score += fp_cost * v
        else:
            score += tp_cost * v
    return score / matching_coo.sum()


def base_edge_penalty_continuous(
    adata1: AnnData,
    adata2: AnnData,
    matching: csr_matrix,
    label: str,
    unit_cost: float,
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

    unit_cost: float
        Label discrepancy is penalized proportionally to this value.
    """
    # Sanity checks
    assert label in adata1.obs, f"Label {label} missing in anndata 1."
    assert label in adata2.obs, f"Label {label} missing in anndata 2."

    n1, n2 = adata1.n_obs, adata2.n_obs
    assert matching.shape == (
        n1,
        n2,
    ), f"Unexpected matching shape, {matching.shape} != {(n1, n2)}"

    if matching.count_nonzero() == 0:
        return 0.0

    label1 = adata1.obs[label].to_numpy()
    label2 = adata2.obs[label].to_numpy()
    matching_coo = matching.tocoo().astype(float)
    penalty = 0.0
    for i1, i2, v in zip(matching_coo.row, matching_coo.col, matching_coo.data):
        penalty += unit_cost * v * abs(label1[i1] - label2[i2])
    return penalty / matching_coo.sum()


def edge_accuracy(
    adata1: AnnData,
    adata2: AnnData,
    matching: csr_matrix,
    label: str,
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
    """
    return base_edge_penalty_discrete(adata1, adata2, matching, label, 1, 0)


def edge_penalty(
    adata1: AnnData,
    adata2: AnnData,
    matching: csr_matrix,
    label: str,
) -> float:
    """
    Computes an accuracy metric for a matching between two datasets whose
    point labels are known. For every point $x matched with $y1 .. $yn,
    its score is given by

    sum_{i=1..n} M(x, yi)*|x-yi|*u / sum_{i=1..n} M(x, yi)

    where M(x, yi) is the matching strength between x and yi, and u is a
    unit penalty cost.

    Then, edge accuracy is given as the average over all points of this
    score.

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
    """
    return base_edge_penalty_continuous(adata1, adata2, matching, label, 1)
