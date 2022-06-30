#!/usr/bin/env python3

import numpy as np

from ot.gromov import fused_gromov_wasserstein
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from typing import Dict, Literal, List, Optional

from ..matching import Matching, _TypeMatchingSet
from ...traits.isprofilable import profile_method
from ...traits.usescommonfeatures import UsesCommonFeatures
from ...traits.usesmetric import UsesMetric
from ...traits.usesreference import UsesReference


class FusedGW(Matching, UsesCommonFeatures, UsesMetric, UsesReference):
    """
    Fused Gromov-Wasserstein-based [1] matching. Embeds the
    ot.gromov.fused_gromov_wasserstein method from POT
    package:

        https://github.com/PythonOT/POT

    It computes a combination of Gromov-Wasserstein and Optimal
    Transport, weighted by a coefficient alpha. Datasets must be
    able to be embedded in a common space, at least pairwisely.

    Parameters
    ----------
    OT_metric: str or callable, default = "sqeuclidean"
        Scipy-compatible metric for optimal transport cost matrix.

    OT_metric_kwargs: dict, default = {}
        Additional metric parameters for OT_metric.

    default_GW_metric: str or callable, default = "sqeuclidean"
        Scipy-compatible default metric for GW computation.

    default_GW_metric_kwargs: dict, default = {}
        Additional default metric parameters for default_GW_metric.

    alpha: float, default = 0.5
        Ratio between optimal transport and Gromov-Wasserstein terms
        in the optimization problem.

    GW_loss: Literal["square_loss", "kl_loss"], default = "square_loss"
        Loss to use in the Gromov-Wasserstein problem. Valid options
        are "square_loss", "kl_loss".

    common_features_mode: Literal["pairwise", "total"]
        Uses pairwise common features, or total common features. Use "total"
        for a small number of datasets, and "pairwise" if the features
        intersection is too small. Ignored if datasets are not in feature
        space.

    References
    ----------
    [1] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain and
        Courty Nicolas “Optimal Transport for structured data with application
        on graphs”, International Conference on Machine Learning (ICML). 2019.
    """

    def __init__(
        self,
        OT_metric: str = "sqeuclidean",
        OT_metric_kwargs: Optional[Dict] = None,
        default_GW_metric: str = "sqeuclidean",
        default_GW_metric_kwargs: Optional[Dict] = None,
        alpha: float = 0.5,
        GW_loss: Literal["square_loss", "kl_loss"] = "square_loss",
        common_features_mode: Literal["pairwise", "total"] = "pairwise",
    ):
        Matching.__init__(self, str_identifier="FUSEDGW")
        UsesCommonFeatures.__init__(self, mode=common_features_mode)
        UsesReference.__init__(self)
        UsesMetric.__init__(self, default_GW_metric, default_GW_metric_kwargs)
        self.OT_metric = OT_metric
        self.OT_metric_kwargs = {} if OT_metric_kwargs is None else OT_metric_kwargs
        self.alpha = alpha
        self.GW_loss = GW_loss

    @profile_method
    def fit(
        self,
        datasets: List[np.ndarray],
    ) -> _TypeMatchingSet:
        """
        Compute optimal transport plan for the FGW problem.
        """
        # Precomputes weights and internal distances
        all_w = [np.ones(X.shape[0]) / X.shape[0] for X in datasets]
        all_metrics = [self.get_metric(i) for i in range(len(datasets))]
        all_C = [
            cdist(Xi, Xi, metric, **kwargs)
            for Xi, (metric, kwargs) in zip(datasets, all_metrics)
        ]
        all_C = [C / C.max() for C in all_C]

        # Compute pairwise FGW
        result: _TypeMatchingSet = {}
        ndatasets = len(datasets)
        reference = self.reference_index
        if reference is None:
            target_indices = np.arange(ndatasets)
        else:
            target_indices = [reference]
        for i, Xi in enumerate(datasets):
            for j in target_indices:
                if (i, j) in result:
                    continue
                Xj = datasets[j]
                Xi_common, Xj_common = self.slice_features(
                    X1=Xi,
                    X2=Xj,
                    idx_1=i,
                    idx_2=j,
                )
                M = cdist(
                    Xi_common,
                    Xj_common,
                    metric=self.OT_metric,
                    *self.OT_metric_kwargs,
                )
                M /= M.max()
                Tij = fused_gromov_wasserstein(
                    M,
                    all_C[i],
                    all_C[j],
                    all_w[i],
                    all_w[j],
                    self.GW_loss,
                    self.alpha,
                )
                result[i, j] = csr_matrix(Tij)
                result[j, i] = csr_matrix(Tij.T)
        return result
