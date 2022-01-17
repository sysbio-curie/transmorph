from typing import Optional, Dict

import numpy as np
from scipy.spatial.distance import cdist


class TData:
    """
    Class that contains the object representing our data.
    In this class, you can find the gene expression matrices and
    also the distance matrix between samples.
    """

    def __init__(
        self,
        X: np.ndarray,
        metric: Optional[str],
        metric_kwarg: Optional[Dict] = {},
        geodesic_distance: Optional[bool] = False,
    ):
        self.X = X
        if metric:
            self.D = cdist(X, X, metric=metric, **metric_kwarg)
