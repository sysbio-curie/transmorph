#!/usr/bin/env python3

from anndata import AnnData
from typing import Dict, List, Literal, Optional

from transmorph.engine import Model
from transmorph.engine.layers import (
    LayerInput,
    LayerTransformation,
    LayerMatching,
    LayerMerging,
    LayerOutput,
)
from transmorph.engine.matching import OT, GW
from transmorph.engine.merging import Barycenter
from transmorph.engine.transforming import CommonFeatures, PCA
from transmorph.utils.anndata_manager import get_total_feature_slices


class TransportCorrection(Model):
    """
    This model performs preprocessing steps, then carries out optimal transport
    between pairs of datasets. It then correctes every dataset with respect to
    a reference dataset chosen by the user using barycentric embedding.

    Parameters
    ----------
    matching: Literal["ot", "gromov"], default = "ot"
        Transportation framework to use, either Optimal Transport (OT) or
        Gromov-Wasserstein (GW). Optimal transport requires batches to be
        embedded in the same space, while Gromov-Wasserstein can work with
        batches embedded in different space at the cost of weaker constraints.

    solver: Literal["exact", "entropic", "unbalanced"], default = "exact"
        Optimization problem formulation.

        - "exact" is available for both OT and Gromov-Wasserstein, and attempts
          to solve exactly the optimization problem. It can scale poorly to large
          datasets (with several tens of thousands of points).
        - "entropic" is available for both OT and Gromov-Wasserstein, and uses
          en entropic regularizer term to be able to use a convex solver. The solution
          is approximate.
        - "unbalanced" is available for OT only, and uses the unbalanced OT formulation
          to deal with datasets with unbalanced class proportions.

    entropy_epsilon: Optional[float], default = None
        If solver is "entropy", allows to tweak the entropy term strength.

    unbalanced_reg: Optional[float]
        Mass conservation regularizer to use in the unbalanced optimal
        transport formulation. The higher, the closer result is from
        constrained optimal transport. The lower, the better the matching
        will be dealing with unbalanced datasets, but convergence will be
        harder. Will be set to 1e-1 by default.

    matching_metric: str, default = "sqeuclidean"
        Metric to use to determine nearest neighbors.

    matching_metric_kwargs: Optional[Dict], default = None
        Additional metric parameters.

    obs_class: Optional[str], default = None
        Provides the AnnData.obs key where sample type is stored. If
        specified, matching edges between samples of different class
        are discarded.

    n_components: int, default = 30
        Number of principal components to use if data dimensionality is
        greater.

    use_feature_space: bool, default = True
        Performs correction in datasets feature space rather than in PC space.

    verbose: bool, default = True
        Logs information in console.

    Example
    -------
    >>> from transmorph.datasets import load_chen_10x
    >>> from transmorph.models import TransportCorrection
    >>> model = TransportCorrection()
    >>> dataset = load_chen_10x()
    >>> model.fit(datasets, reference=dataset['P01'])
    """

    def __init__(
        self,
        matching: Literal["ot", "gromov"] = "ot",
        solver: Literal["exact", "entropic", "partial", "unbalanced"] = "exact",
        entropy_epsilon: Optional[float] = None,
        unbalanced_reg: Optional[float] = None,
        matching_metric: str = "sqeuclidean",
        matching_metric_kwargs: Optional[Dict] = None,
        obs_class: Optional[str] = None,
        n_components: int = 30,
        use_feature_space: bool = False,
        verbose: bool = True,
    ) -> None:
        from .. import settings

        if verbose:
            settings.verbose = "INFO"
        else:
            settings.verbose = "WARNING"

        # Loading algorithms
        if matching == "ot":
            if solver == "exact":
                solver_ot = "emd"
            elif solver == "entropic":
                solver_ot = "sinkhorn"
            else:
                solver_ot = "unbalanced"
            matching_alg = OT(
                solver=solver_ot,
                metric=matching_metric,
                metric_kwargs=matching_metric_kwargs,
                common_features_mode="total",
                sinkhorn_reg=entropy_epsilon,
                unbalanced_reg=unbalanced_reg,
            )
        elif matching == "gromov":
            if solver == "exact":
                solver_gw = "gw"
            elif solver == "entropic":
                solver_gw = "entropic_gw"
            else:
                raise ValueError(f"Solver {solver} not found for GW matching.")
            matching_alg = GW(
                optimizer=solver_gw,
                default_metric=matching_metric,
                default_metric_kwargs=matching_metric_kwargs,
                GW_loss="square_loss",
                entropy_epsilon=entropy_epsilon,
            )
        else:
            raise ValueError(
                f"Unrecognized matching: {matching}. Expected 'mnn' or 'bknn'."
            )
        merging = Barycenter()

        # Initializing layers
        linput = LayerInput()

        ltransform_features = LayerTransformation()
        if matching == "ot":
            ltransform_features.add_transformation(CommonFeatures())

        ltransform_dimred = LayerTransformation()
        ltransform_dimred.add_transformation(PCA(n_components=n_components))

        lmatching = LayerMatching(matching=matching_alg, obs_class=obs_class)

        lmerging = LayerMerging(merging=merging)

        loutput = LayerOutput()

        # Building model
        linput.connect(ltransform_features)
        ltransform_features.connect(ltransform_dimred)
        ltransform_dimred.connect(lmatching)
        lmatching.connect(lmerging)

        lmerging.connect(loutput)

        self.use_feature_space = use_feature_space
        if use_feature_space:
            lmerging.embedding_reference = ltransform_features
        else:
            lmerging.embedding_reference = ltransform_dimred

        Model.__init__(self, input_layer=linput, str_identifier="MNN_CORRECTION")

    def transform(
        self,
        datasets: List[AnnData],
        reference: AnnData,
        use_representation: Optional[str] = None,
        output_representation: Optional[str] = None,
    ) -> None:
        """
        Carries out the model on a list of AnnData objects. Writes the result in
        .obsm fields.

        Parameters
        ----------
        datasets: List[AnnData]
            List of anndata objects, must have at least one common feature.

        reference: AnnData
            Reference dataset for the correction.

        use_representation: Optional[str]
            .obsm to use as input.

        output_representation: str
            .obsm destination key, "transmorph" by default.
        """
        if isinstance(datasets, Dict):
            datasets = list(datasets.values())
        self.fit(
            datasets,
            reference=reference,
            use_representation=use_representation,
            output_representation=output_representation,
        )
        if self.use_feature_space:
            self.embedding_features = datasets[0].var_names[
                get_total_feature_slices(datasets)[0]
            ]
