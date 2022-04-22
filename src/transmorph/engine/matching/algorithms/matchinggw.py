#!/usr/bin/env python3

import numpy as np

from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from typing import Any, Dict, List, Literal, Optional
from ot.gromov import gromov_wasserstein, entropic_gromov_wasserstein

from ..matching import Matching, _TypeMatchingSet
from ...traits.hasmetadata import HasMetadata
from ...traits.isprofilable import profile_method
from ...traits.usesmetric import UsesMetric


class MatchingGW(Matching, UsesMetric, HasMetadata):
    """
    Gromov-Wasserstein-based matching. Embeds the gromov_wasserstein class of
    methods from POT:

        https://github.com/PythonOT/POT

    Gromov-Wasserstein(GW) computes a transport plan between two distributions,
    and does not require them to be defined in the same space. It rather use
    relative topology of each distribution in its own metric space to define a
    cost that assumes similar locations to have similar relative positions with
    respect to the other regions. This combinatorial cost is typically more
    expansive than the optimal transport alternative, but comes very handy when
    a ground cost is difficult (or impossible) to compute between distributions.

    Parameters
    ----------
    optimizer: Literal["GW", "GWEntropic"], default = "GW"
        Uses the exact (GW) or entropy regularized (GWEntropic) problem
        formulation.

    default_metric: str or callable, default = "sqeuclidean"
        Scipy-compatible metric to use for datasets in which metric
        is not specified as a backup.

    default_metric_kwargs: dict, default = {}
        Additional metric parameters for backup metric.

    GW_loss: str, default = "square_loss"
        Loss to use in the Gromov-Wasserstein problem. Valid options
        are "square_loss", "kl_loss".

    entropy_epsilon: Optional[float]
        If optimizer is GWEntropic, this is the regularization parameter.
        Decreasing it will improve result quality, but it will decrease
        convergence speed.

    max_iter: int, default = 1e6
        Maximum number of iterations to solve the optimization problem.

    References
    ----------
    [1] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    [2] Mémoli, Facundo. Gromov–Wasserstein distances and the
        metric approach to object matching. Foundations of computational
        mathematics 11.4 (2011): 417-487.
    """

    def __init__(
        self,
        optimizer: Literal["gw", "entropic_gw"] = "gw",
        default_metric: str = "sqeuclidean",
        default_metric_kwargs: Optional[Dict] = None,
        GW_loss: str = "square_loss",
        entropy_epsilon: Optional[float] = None,
        max_iter: int = int(1e6),
    ):
        Matching.__init__(self, str_identifier="GW")
        UsesMetric.__init__(self)
        default_metric_kwargs = (
            {} if default_metric_kwargs is None else default_metric_kwargs
        )
        default_metadata = {
            "metric": default_metric,
            "metric_kwargs": default_metric_kwargs,
        }
        HasMetadata.__init__(self, default_metadata)
        assert optimizer in ("gw", "entropic_gw"), f"Unknown optimizer: {optimizer}."
        self.optimizer = optimizer
        self.GW_loss = GW_loss
        if entropy_epsilon is not None and optimizer == "gw":
            self.warn("Epsilon specified has no effect on gw optimizer.")
        if entropy_epsilon is None:
            entropy_epsilon = 1e-2
        self.entropy_epsilon = entropy_epsilon
        self.max_iter = int(max_iter)

    def retrieve_metadatata(self, adata: AnnData) -> Dict[str, Any]:
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
        # Selects optimizer
        if self.optimizer == "gw":
            optimizer = gromov_wasserstein
            kwargs = {"numItermax": self.max_iter}
        elif self.optimizer == "entropic_gw":
            optimizer = entropic_gromov_wasserstein
            kwargs = {"epsilon": self.entropy_epsilon, "max_iter": self.max_iter}
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}.")
        # Compute pairwise GW
        ndatasets = len(datasets)
        result: _TypeMatchingSet = {}
        for i in range(ndatasets):
            for j in range(i + 1, ndatasets):
                Tij = optimizer(
                    C1=all_C[i],
                    C2=all_C[j],
                    p=all_w[i],
                    q=all_w[j],
                    loss_fun=self.GW_loss,
                    **kwargs,
                )
                result[i, j] = csr_matrix(Tij)
                result[j, i] = csr_matrix(Tij.T)
        return result
