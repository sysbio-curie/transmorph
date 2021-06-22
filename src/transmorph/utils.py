#!/usr/bin/env python3

import numpy as np

from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

def tdist(X, Y=None, normalize=True, n_comps=-1, metric='sqeuclidean'):
    """
    Utility/QoL function embedding scipy's cdist. Returns the cost matrix
    between two column-normalized dataset, for a given metric. If needed,
    a dimensionality reduction technique can be applied first to reduce
    the curse of dimensionality.

    Parameters:
    -----------

    X: (n,d) np.ndarray
        First dataset in matrix form.

    Y: (m,d) np.ndarray, default=None
        Second dataset in matrix form. If None, Y=X.

    normalize: bool, default=True
        STD-normalizes each dataset column-wise.

    n_comps: int, default=-1
        Number of PCs to compute before estimating distance. If -1,
        raw datasets are used.

    metric: str, default='euclidean'
        Metric argument for scipy.spatial.distance.cdist.
    """
    if Y is None:
        Y = X

    assert X.shape[1] == Y.shape[1], "Incompatible dimensions: %i vs %i" % (
        X.shape[1], Y.shape[1]
    )

    # Columns normalization
    if normalize:
        X_std = X.std(axis=0)
        X_std[X_std == 0] = 1
        X = X / X_std
        Y_std = Y.std(axis=0)
        Y_std[Y_std == 0] = 1
        Y = Y / Y_std

    # Dimension reduction if needed
    if n_comps != -1:
        pca = PCA(n_comps=n_comps)
        XY = np.concat( (X, Y), axis=0 )
        XY_red = pca.fit_transform(XY)
        X, Y = XY_red[:len(X),:], XY_red[len(X):,]
    elif X.shape[1] > 30:
        print("Warning: High dimensional data. Consider using a lower\
        dimensional representation.")

    return cdist(X, Y, metric=metric)
