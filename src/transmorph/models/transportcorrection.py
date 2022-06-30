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
from transmorph.engine.matching import OT
from transmorph.engine.merging import Barycenter
from transmorph.engine.transforming import CommonFeatures, PCA
from transmorph.utils.anndata_manager import get_total_feature_slices


class TransportIntegration(Model):
    """
    Performs OT, then barycentric merging.

    Parameters
    ----------
    """

    def __init__(
        self,
        matching: Literal["ot", "gromov"] = "ot",
        solver: Literal["emd", "sinkhorn", "partial", "unbalanced"] = "emd",
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
            matching_alg = OT(
                solver=solver,
                metric=matching_metric,
                metric_kwargs=matching_metric_kwargs,
                common_features_mode="total",
            )
        else:
            raise ValueError(
                f"Unrecognized matching: {matching}. Expected 'mnn' or 'bknn'."
            )
        merging = Barycenter()

        # Initializing layers
        linput = LayerInput()

        ltransform_features = LayerTransformation()
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
        """
        self.fit(
            datasets,
            reference=reference,
            use_representation=use_representation,
        )
        if self.use_feature_space:
            self.embedding_features = datasets[0].var_names[
                get_total_feature_slices(datasets)[0]
            ]
