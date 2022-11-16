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
    This model performs preprocessing steps, then carries out mutual nearest
    neighbors (MNN) between pairs of datasets. It then corrects every dataset
    with respect to a reference dataset chosen by the user.

    Parameters
    ----------
    matching: Literal["mnn", "bknn"], default = "bknn"
        Nearest neighbors matching to use, either Mutual Nearest Neighbors
        (MNN) or Batch k-Nearest Neighbors (BKNN). For a given number of
        neighbors $k$, here is the subtlety between both algorithms.

        - In MNN, points $x_i$ from batch $X$ and $y_j$ from batch $Y$ are
          matched if $x_i$ belongs to the $k$ nearest neighbors of $y_j$ in
          $X$, and vice-versa.
        - In BKNN, point $x_i$ is matched with point $y_j$ if $y_j$ belongs
          to the $k$ nearest neighbors of $x_i$ in $Y$.

    matching_n_neighbors: int, default = None
        Number of neighbors to use for the mutual nearest neighbors step. If
        None is provided, it is determined automatically.

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

    lc_n_neighbors: int, default = 10
        Number of neighbors to use for linear correction neighbors graph.

    use_feature_space: bool, default = True
        Performs correction in datasets feature space rather than in PC space.

    verbose: bool, default = True
        Logs information in console.

    Example
    -------
    >>> from transmorph.datasets import load_chen_10x
    >>> from transmorph.models import MNNCorrection
    >>> model = MNNCorrection()
    >>> dataset = load_chen_10x()
    >>> model.fit(datasets, reference=dataset['P01'])
    """

    def __init__(
        self,
        matching: Literal["mnn", "bknn"] = "bknn",
        matching_n_neighbors: Optional[int] = None,
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
            )
        elif matching == "bknn":
            if matching_n_neighbors is None:
                matching_n_neighbors = 10
            matching_alg = BKNN(
                metric=matching_metric,
                metric_kwargs=matching_metric_kwargs,
                n_neighbors=matching_n_neighbors,
                common_features_mode="total",
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
