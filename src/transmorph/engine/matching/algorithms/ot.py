#!/usr/bin/env python3

import numpy as np

from ot import emd
from ot.partial import partial_wasserstein
from ot.bregman import sinkhorn_stabilized
from ot.unbalanced import sinkhorn_stabilized_unbalanced
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from typing import Dict, List, Literal, Optional, Union

from ..matching import Matching, _TypeMatchingSet
from ...traits.isprofilable import profile_method
from ...traits.usescommonfeatures import UsesCommonFeatures
from ...traits.usesreference import UsesReference


class OT(Matching, UsesCommonFeatures, UsesReference):
    """
    Optimal transport-based matching. Wraps a selection of optimal
    transport-derived methods from POT package:

        https://github.com/PythonOT/POT

    Optimal transport is an optimization problem seeking to move
    mass from a source ptobability distribution to a target probability
    distribution embedded in the same metric space at minimal cost,
    where cost is both proportional to mass moved and distance moved.
    We assume optimal transport can yield relevant matchings, being
    more likely to move samples of one class towards samples of the
    same class in the other dataset. There exists many variants of the
    optimal transport problem, to help scalability or deal with
    dataset class imbalance.

    We wrapped four solvers:

    - Earth Mover's Distance (EMD) which solves the exact
      optimal transport problem [1].

    - Sinkhorn-Knopp solver for the entropy-regularized
      optimal transport problem [2], which can help for larger
      datasets. We use a log-stabilized implementation for
      more reliable convergence [3, 4].

    - Partial optimal transport solver, which adds dummy
      points to each datasets [5]. These dummy points are supposed
      to attract dataset-specific classes, but need to have their
      mass parametrized. This mass can be parametrized as a float,
      or on a per-dataset basis.

    - Unbalanced solver, which introduces mass conservation as
      a penalty instead of a constraint [6]. This relaxation allows
      a better handling of class imbalanced datasets, but is
      typically longer and less likely to converge to a good optimal
      transport plan in practice.

    Parameters
    ----------
    solver: Literal["emd", "sinkhorn", "partial", "unbalanced"], default = "emd"
        Solver to use.

    metric: str, default = "sqeuclidean"
        Scipy-compatible metric to compute cost matrix.

    metric_kwargs: Optional[Dict]
        Additional metric parameters if necessary.

    common_features_mode: Literal["pairwise", "total"]
        Uses pairwise common features, or total common features. Use "total"
        for a small number of datasets, and "pairwise" if the features
        intersection is too small.

    sinkhorn_reg: Optional[float]
        Entropy regularizer to use for Sinkhorn-Knopp-based solvers. Increasing
        it will increase rate of convergence, but decrease optimal transport
        estimation accuracy. If not specified, uses 10^-2.

    partial_transport_mass: Optional[Union[float, List[float]]]
        Fraction of mass attributed to dummy points, must be between 0 and 1.
        If a float is provided, will be used for all datasets. If a list of
        float is provided, each dataset will have its own mass.

    partial_n_dummies: Optional[Union[float, List[float]]]
        Number of dummies to use in the partial formulation.
        If an integer is provided, will be used for all datasets. If a list of
        integers is provided, each dataset will have its own n dummies.

    unbalanced_reg: Optional[float]
        Mass conservation regularizer to use in the unbalanced optimal
        transport formulation. The higher, the closer result is from
        constrained optimal transport. The lower, the better the matching
        will be dealing with unbalanced datasets, but convergence will be
        harder. Will be set to 1e-1 by default.

    max_iter: int, default = 1e6
        Maximum number of iterations to solve the optimization problem.

    References
    ----------
    [1] Bonneel, N., Van De Panne, M., Paris, S., & Heidrich, W. (2011, December).
        Displacement interpolation using Lagrangian mass transport. In ACM Transactions
        on Graphics (TOG) (Vol. 30, No. 6, p. 158). ACM.

    [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport,
        Advances in Neural Information Processing Systems (NIPS) 26, 2013

    [3] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms for Entropy
        Regularized Transport Problems. arXiv preprint arXiv:1610.06519.

    [4] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016). Scaling
        algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.

    [5] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G. (2015).
        Iterative Bregman projections for regularized transportation problems.
        SIAM Journal on Scientific Computing, 37(2), A1111-A1138.

    [6] Frogner C., Zhang C., Mobahi H., Araya-Polo M., Poggio T. : Learning with a
        Wasserstein Loss, Advances in Neural Information Processing Systems (NIPS) 2015
    """

    def __init__(
        self,
        solver: Literal["emd", "sinkhorn", "partial", "unbalanced"] = "emd",
        metric: str = "sqeuclidean",
        metric_kwargs: Optional[Dict] = None,
        common_features_mode: Literal["pairwise", "total"] = "pairwise",
        sinkhorn_reg: Optional[float] = None,
        partial_transport_mass: Optional[Union[float, List[float]]] = None,
        partial_n_dummies: Optional[Union[int, List[int]]] = None,
        unbalanced_reg: Optional[float] = None,
        max_iter: int = int(1e6),
    ):
        Matching.__init__(self, str_identifier="OT")
        UsesCommonFeatures.__init__(self, mode=common_features_mode)
        UsesReference.__init__(self)

        # Sanity checks
        assert solver in (
            "emd",
            "sinkhorn",
            "partial",
            "unbalanced",
        ), f"Unknown solver: {solver}."

        # General parameters
        self.solver = solver
        self.metric = metric
        self.metric_kwargs = {} if metric_kwargs is None else metric_kwargs
        self.max_iter = int(max_iter)

        # Sinkhorn parameters
        if sinkhorn_reg is None:
            sinkhorn_reg = 1e-2
        self.sinkhorn_reg = sinkhorn_reg

        # Partial parameters
        if partial_transport_mass is None:
            partial_transport_mass = 0.5
        if isinstance(partial_transport_mass, float):
            assert 0.0 <= partial_transport_mass <= 1.0, (
                "Partial transport mass must be between 0.0 and 1.0, "
                f"found {partial_transport_mass}."
            )
        if isinstance(partial_transport_mass, List):
            for pm in partial_transport_mass:
                assert 0.0 <= pm <= 1.0, (
                    "Partial transport mass must be between 0.0 and 1.0, "
                    f"found {pm}."
                )
        if partial_n_dummies is None:
            partial_n_dummies = 1
        self.partial_transport_mass = partial_transport_mass
        self.partial_n_dummies = partial_n_dummies

        # Unbalanced parameters
        if unbalanced_reg is None:
            unbalanced_reg = 1e-1
        self.unbalanced_reg = unbalanced_reg

    @profile_method
    def fit(
        self,
        datasets: List[np.ndarray],
        reference: Optional[int] = None,
    ) -> _TypeMatchingSet:
        """
        Computes OT between pairs of datasets with the right solver.
        """
        ndatasets = len(datasets)
        results: _TypeMatchingSet = {}
        ndatasets = len(datasets)
        reference = self.reference_index
        if reference is None:
            target_indices = np.arange(ndatasets)
        else:
            target_indices = [reference]
        for i in range(ndatasets):
            for j in target_indices:
                if (i, j) in results:
                    continue
                kwargs = {}
                if self.solver == "emd":
                    solver = emd
                elif self.solver == "sinkhorn":
                    solver = sinkhorn_stabilized
                    kwargs = {"reg": self.sinkhorn_reg}
                elif self.solver == "partial":
                    solver = partial_wasserstein
                    if isinstance(self.partial_transport_mass, List):
                        partial_mass = self.partial_transport_mass[i]
                    else:
                        partial_mass = self.partial_transport_mass
                    if isinstance(self.partial_n_dummies, List):
                        partial_dummies = self.partial_n_dummies[i]
                    else:
                        partial_dummies = self.partial_n_dummies
                    kwargs = {
                        "m": partial_mass,
                        "nb_dummies": partial_dummies,
                    }
                elif self.solver == "unbalanced":
                    solver = sinkhorn_stabilized_unbalanced
                    kwargs = {"reg": self.sinkhorn_reg, "reg_m": self.unbalanced_reg}
                else:
                    raise ValueError(f"Unknown solver: {self.solver}")
                Xi, Xj = datasets[i], datasets[j]
                ni, nj = Xi.shape[0], Xj.shape[0]
                wi, wj = np.ones(ni) / ni, np.ones(nj) / nj
                Xi, Xj = self.slice_features(X1=Xi, X2=Xj, idx_1=i, idx_2=j)
                M = cdist(Xi, Xj, metric=self.metric, **self.metric_kwargs)
                M /= M.max()
                Tij = solver(a=wi, b=wj, M=M, numItermax=self.max_iter, **kwargs)
                results[i, j] = csr_matrix(Tij)
                results[j, i] = csr_matrix(Tij.T)
        return results
