"""Implementation based on https://github.com/slowkow/harmonypy/blob/master/harmonypy
/lisi.py """

import numpy as np

from ..utils import sort_sparse_matrix
from ..utils.graph import nearest_neighbors


def compute_lisi(X: np.ndarray, sample_labels: np.ndarray, perplexity: float = 30.0):
    """
    Compute the Local Inverse Simpson Index (LISI) for each column in metadata.
    LISI is a statistic computed for each item (row) in the data matrix X.
    The following example may help to interpret the LISI values.
    Suppose one of the columns in metadata is a categorical variable with 3 categories.
        - If LISI is approximately equal to 3 for an item in the data matrix,
          that means that the item is surrounded by neighbors from all 3
          categories.
        - If LISI is approximately equal to 1, then the item is surrounded by
          neighbors from 1 category.

    The LISI statistic is useful to evaluate whether multiple datasets are
    well-integrated by algorithms such as Harmony [1].
    [1]: Korsunsky et al. 2019 doi: 10.1038/s41592-019-0619-0
    Parameters
    ----------
    X: Matrix of shape (n_samples, d) where M[i, j] with the coordinates of samples.
    sample_labels: Matrix of shape (n_samples, k) where M[i, j] is the label of
        point $i for label type $j.
    perplexity
        Parameters for lisi.
    Returns
    -------
        A Matrix of shape (n_samples, k) with estimation of lisi for each point
        and each label.
    """
    n_cells = sample_labels.shape[0]
    n_labels = sample_labels.shape[1]
    # We need at least 3 * n_neigbhors to compute the perplexity
    connectivity = nearest_neighbors(
        X, n_neighbors=int(perplexity * 3), include_self_loops=False
    )
    indices, distances = sort_sparse_matrix(connectivity)
    # Save the result
    lisi_array = np.zeros((n_cells, n_labels))
    for i in range(n_labels):
        n_categories = len(np.unique(sample_labels[:, i]))
        simpson = compute_simpson(
            distances.T, indices.T, sample_labels[:, i], n_categories, perplexity
        )
        lisi_array[:, i] = 1 / simpson
    return lisi_array


def compute_simpson(
    distances: np.ndarray,
    indices: np.ndarray,
    labels: np.ndarray,
    perplexity: float,
    tol: float = 1e-5,
):
    n = distances.shape[1]
    simpson = np.zeros(n)
    logU = np.log(perplexity)
    # Loop through each cell.
    for i in range(n):
        beta = 1
        betamin = -np.inf
        betamax = np.inf
        # Compute Hdiff
        P = np.exp(-distances[:, i] * beta)
        P_sum = np.sum(P)
        if P_sum == 0:
            H = 0
            P = np.zeros(distances.shape[0])
        else:
            H = np.log(P_sum) + beta * np.sum(distances[:, i] * P) / P_sum
            P = P / P_sum
        Hdiff = H - logU
        n_tries = 50
        for t in range(n_tries):
            # Stop when we reach the tolerance
            if abs(Hdiff) < tol:
                break
            # Update beta
            if Hdiff > 0:
                betamin = beta
                if not np.isfinite(betamax):
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if not np.isfinite(betamin):
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2
            # Compute Hdiff
            P = np.exp(-distances[:, i] * beta)
            P_sum = np.sum(P)
            if P_sum == 0:
                H = 0
                P = np.zeros(distances.shape[0])
            else:
                H = np.log(P_sum) + beta * np.sum(distances[:, i] * P) / P_sum
                P = P / P_sum
            Hdiff = H - logU
        # distancesefault value
        if H == 0:
            simpson[i] = -1
        # Simpson's index
        for label_category in np.unique(labels):
            ix = indices[:, i]
            q = labels[ix] == label_category
            if np.any(q):
                P_sum = np.sum(P[q])
                simpson[i] += P_sum * P_sum
    return simpson
