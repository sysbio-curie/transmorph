#!/usr/bin/env python3

import numpy as np

from anndata import AnnData
from ot.gromov import fused_gromov_wasserstein
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from typing import Any, Dict, Hashable, Literal, List, Optional

from ..matching import Matching, _TypeMatchingSet
from ...traits.hasmetadata import HasMetadata
from ...traits.isprofilable import profile_method
from ...traits.usescommonfeatures import UsesCommonFeatures
from ...traits.usesmetric import UsesMetric


class FusedGW(Matching, UsesCommonFeatures, HasMetadata, UsesMetric):
    """
    Fused Gromov-Wasserstein-based matching. Embeds the
    ot.gromov.fused_gromov_wasserstein method from POT:

        https://github.com/PythonOT/POT

    It computes a combination of Gromov-Wasserstein and Optimal
    Transport, weighted by a coefficient alpha.
    Both datasets need to be in the same
    space in order to compute a cost matrix.

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

    GW_loss: str, default = "square_loss"
        Loss to use in the Gromov-Wasserstein problem. Valid options
        are "square_loss", "kl_loss".

    common_features_mode: Literal["pairwise", "total"]
        Uses pairwise common features, or total common features. Use "total"
        for a small number of datasets, and "pairwise" if the features
        intersection is too small.
    """

    def __init__(
        self,
        OT_metric: str = "sqeuclidean",
        OT_metric_kwargs: Optional[Dict] = None,
        default_GW_metric: str = "sqeuclidean",
        default_GW_metric_kwargs: Optional[Dict] = None,
        alpha: float = 0.5,
        GW_loss: str = "square_loss",
        common_features_mode: Literal["pairwise", "total"] = "pairwise",
    ):
        Matching.__init__(self, str_identifier="FUSEDGW")
        UsesCommonFeatures.__init__(self, mode=common_features_mode)
        UsesMetric.__init__(self)
        default_metric_kwargs = (
            {} if default_GW_metric_kwargs is None else default_GW_metric_kwargs
        )
        default_metadata = {
            "metric": default_GW_metric,
            "metric_kwargs": default_metric_kwargs,
        }
        HasMetadata.__init__(self, default_metadata)
        self.OT_metric = OT_metric
        self.OT_metric_kwargs = {} if OT_metric_kwargs is None else OT_metric_kwargs
        self.alpha = alpha
        self.GW_loss = GW_loss

    def retrieve_metadatata(self, adata: AnnData) -> Dict[Hashable, Any]:
        """
        Retrieves custom metric contained in AnnData if any.
        """
        metric_and_kwargs = UsesMetric.get_metric(adata)
        if metric_and_kwargs is None:
            return {}
        else:
            metric, metric_kwargs = metric_and_kwargs
        metadata = {}
        if metric is not None:
            metadata["metric"] = metric
        if metric_kwargs is not None:
            metadata["metric_kwargs"] = metric_kwargs
        return metadata

    @profile_method
    def fit(self, datasets: List[np.ndarray]) -> _TypeMatchingSet:
        """
        Compute optimal transport plan for the FGW problem.
        TODO: specific strategy if reference is set
        """
        # Precomputes weights and internal distances
        all_w = [np.ones(X.shape[0]) / X.shape[0] for X in datasets]
        all_C = [
            cdist(
                Xi,
                Xi,
                self.get_metadata(i, "metric"),
                **self.get_metadata(i, "metric_kwargs"),
            )
            for i, Xi in enumerate(datasets)
        ]
        all_C = [C / C.max() for C in all_C]

        # Compute pairwise FGW
        result: _TypeMatchingSet = {}
        for i, Xi in enumerate(datasets):
            for j, Xj in enumerate(datasets):
                if j >= i:
                    continue
                Xi_common, Xj_common = self.slice_features(
                    X1=Xi, X2=Xj, idx_1=i, idx_2=j
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
