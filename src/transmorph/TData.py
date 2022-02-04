from typing import Dict

import numpy as np


class TData:
    """
    Wrapper for a vectorized dataset.

    Parameters
    ----------
    X: np.ndarray
        Vectorized dataset of shape (n_samples, n_features).

    metadata: Dict, default = {}
        Additional key->data information to provide, necessary for
        some matchings.

    Example
    -------
    >>> import numpy as np
    >>> import transmorph as tr
    >>> X = np.ndarray([[2, 1], [1, 1], [-1, 3]])
    >>> metadata = {"type": ["dog", "dog", "cat"]}
    >>> td = tr.TData(X, metadata=metadata)
    """

    def __init__(self, X: np.ndarray, metadata: Dict = {}):
        self.X = X
        self.metadata = metadata
        self.shape = self.X.shape

    def __str__(self):
        return f"<TData {self.shape}>"
