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
from transmorph.engine.matching import BKNN, MNN
from transmorph.engine.merging import LinearCorrection
from transmorph.engine.transforming import CommonFeatures, PCA
from transmorph.utils.anndata_manager import get_total_feature_slices


class MNNCorrection(Model):
    """
    Performs MNN, then linear correction given selected anchors.

    Parameters
    ----------
    mnn_n_neighbors: int, default = 30
        Number of neighbors to use for the mutual nearest neighbors step.

    mnn_metric: str, default = "sqeuclidean"
        Metric to use during MNN step.

    mnn_kwargs: Optional[Dict], default = None
        Additional metric parameters.

    n_components: int, default = 30
        Number of principal components to use if data dimensionality is
        greater.

    use_subsampling: bool, default = False
        Run MNN and LISI on a subsample of points to spare performance.
        Useful for large datasets.

    lc_n_neighbors: int, default = 10
        Number of neighbors to use for inferrence of correction vectors
        in linear correction. The higher, the easier samples are corrected
        but the more approximate it is.

    use_feature_space: bool, default = True
        Do the integration in feature space. Otherwise, do it in PC space.
        Increases computational complexity.

    verbose: bool, default = True
        Logs runtime information in console.
    """

    def __init__(
        self,
        matching: Literal["mnn", "bknn"] = "bknn",
        matching_n_neighbors: int = 30,
        matching_metric: str = "sqeuclidean",
        matching_metric_kwargs: Optional[Dict] = None,
        obs_class: Optional[str] = None,
        n_components: int = 30,
        lc_n_neighbors: int = 10,
        use_feature_space: bool = True,
        verbose: bool = True,
    ) -> None:
        from .. import settings

        if verbose:
            settings.verbose = "INFO"
        else:
            settings.verbose = "WARNING"

        # Loading algorithms
        if matching == "mnn":
            if matching_n_neighbors is None:
                matching_n_neighbors = 30
            matching_alg = MNN(
                metric=matching_metric,
                metric_kwargs=matching_metric_kwargs,
                n_neighbors=matching_n_neighbors,
                common_features_mode="total",
                solver="auto",
            )
        elif matching == "bknn":
            if matching_n_neighbors is None:
                matching_n_neighbors = 10
            matching_alg = BKNN(
                metric=matching_metric,
                metric_kwargs=matching_metric_kwargs,
                n_neighbors=matching_n_neighbors,
                common_features_mode="total",
                solver="auto",
            )
        else:
            raise ValueError(
                f"Unrecognized matching: {matching}. Expected 'mnn' or 'bknn'."
            )
        merging = LinearCorrection(n_neighbors=lc_n_neighbors)

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
        if isinstance(datasets, Dict):
            datasets = list(datasets.values())
        self.fit(
            datasets,
            reference=reference,
            use_representation=use_representation,
        )
        if self.use_feature_space:
            self.embedding_features = datasets[0].var_names[
                get_total_feature_slices(datasets)[0]
            ]
