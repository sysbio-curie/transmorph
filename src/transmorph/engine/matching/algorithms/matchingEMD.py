#!/usr/bin/env python3

import numpy as np

from ot import emd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from typing import Dict, List, Literal, Optional, Tuple

from transmorph.engine.matching import MatchingABC
from transmorph.engine.traits import UsesCommonFeatures


class MatchingEMD(MatchingABC, UsesCommonFeatures):
    """
    Earth Mover's Distance-based matching. Embeds the ot.emd
    method from POT:

        https://github.com/PythonOT/POT

    ot.emd solves exactly the earth mover's distance problem using
    a C-accelerated backend. Both datasets need to be in the same
    space in order to compute a cost matrix.

    Parameters
    ----------
    metric: str or callable, default = "sqeuclidean"
        Scipy-compatible metric.

    metric_kwargs: dict, default = {}
        Additional metric parameters.

    subsampling: SubsamplingABC, default = None
        Subsampling scheme to apply before computing the matching,
        can be very helpful when dealing with large datasets.

    max_iter: int, default = 1e6
        Maximum number of iterations to solve the optimization problem.

    common_features_mode: Literal["pairwise", "total"]
        Uses pairwise common features, or total common features. Use "total"
        for a small number of datasets, and "pairwise" if the features
        intersection is too small.
    """

    def __init__(
        self,
        metric: str = "sqeuclidean",
        metric_kwargs: Optional[Dict] = None,
        max_iter: int = int(1e6),
        common_features_mode: Literal["pairwise", "total"] = "pairwise",
    ):
        MatchingABC.__init__(self, str_type="MATCHING_EMD")
        UsesCommonFeatures.__init__(self, mode=common_features_mode)
        self.metric = metric
        self.metric_kwargs = {} if metric_kwargs is None else metric_kwargs
        self.max_iter = int(max_iter)

    def check_input(self, datasets: List[np.ndarray]) -> None:
        """
        Does nothing.
        """
        pass

    def fit(self, datasets: List[np.ndarray]) -> Dict[Tuple[int, int], csr_matrix]:
        """
        Simply gathers datasets and compute exact optimal transport.
        """
        ndatasets = len(datasets)
        results = {}
        for i in range(ndatasets):
            for j in range(i + 1, ndatasets):
                Xi, Xj = datasets[i], datasets[j]
                ni, nj = Xi.shape[0], Xj.shape[0]
                wi, wj = np.ones(ni) / ni, np.ones(nj) / nj
                Xj, Xj = self.slice_features(Xi, Xj, i, j)
                M = cdist(Xi, Xj, metric=self.metric, **self.metric_kwargs)
                M /= M.max()
                Tij = emd(wi, wj, M, numItermax=self.max_iter)
                results[i, j] = csr_matrix(Tij)
                results[j, i] = csr_matrix(Tij.T)
        return results
