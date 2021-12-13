#!/usr/bin/env python3

from ot import emd
from scipy.spatial.distance import cdist

from .matchingABC import MatchingABC
from typing import Callable, Union

import numpy as np

from transmorph.TData import TData


class MatchingEMD(MatchingABC):
    """
    Earth Mover's Distance-based matching. Embeds the ot.emd
    method from POT:

        https://github.com/PythonOT/POT

    ot.emd solves exactly the earth mover's distance problem using
    a C-accelerated backend. Both datasets need to be in the same
    space in order to compute a cost matrix.
    TODO: allow the user to define a custom callable metric?

    Parameters
    ----------
    metric: str or callable, default = "sqeuclidean"
        Scipy-compatible metric.

    metric_kwargs: dict, default = {}
        Additional metric parameters.

    max_iter: int, default = 1e6
        Maximum number of iterations to solve the optimization problem.

    use_sparse: boolean, default = True
        Save matching as sparse matrices.
    """

    def __init__(
        self,
        metric: Union[str, Callable] = "sqeuclidean",
        metric_kwargs: dict = {},
        max_iter: int = int(1e6),
        use_sparse: bool = True,
    ):
        MatchingABC.__init__(self, use_sparse=use_sparse)
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.max_iter = int(max_iter)

    def _match2(self, t1: TData, t2: TData):
        n1, n2 = t1.X.shape[0], t2.X.shape[0]
        w1, w2 = np.ones(n1) / n1, np.ones(n2) / n2
        M = cdist(t1.X, t2.X, metric=self.metric, **self.metric_kwargs)
        M /= M.max()
        return emd(w1, w2, M, numItermax=self.max_iter)
