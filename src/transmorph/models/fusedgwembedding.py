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
from transmorph.engine.matching import FusedGW
from transmorph.engine.merging import GraphEmbedding
from transmorph.engine.transforming import CommonFeatures, PCA


class FusedGWEmbedding(Model):
    """

    Parameters
    ----------
    matching_metric: str, default = "sqeuclidean"
        Metric to use to determine nearest neighbors.

    matching_metric_kwargs: Optional[Dict], default = None
        Additional metric parameters.

    alpha: float, default = 0.5
        Ratio between optimal transport and Gromov-Wasserstein terms
        in the optimization problem.

    GW_loss: Literal["square_loss", "kl_loss"], default = "square_loss"
        Loss to use in the Gromov-Wasserstein problem. Valid options
        are "square_loss", "kl_loss".

    obs_class: Optional[str], default = None
        Provides the AnnData.obs key where sample type is stored. If
        specified, matching edges between samples of different class
        are discarded.

    embedding_optimizer: Literal["umap", "mde"], default = "umap"
        Graph embedding algorithm to use.

    embedding_n_neighbors: int, default = 10
        Target number of edges per point in the graph to embed. Only
        this number of most significant edges are conserved. Try to
        keep this number greater than dataset number.

    embedding_dimension: int, default = 2
        Number of dimensions in the final embedding.

    pca_n_components: int, default = 30
        Number of principal components to use if data dimensionality is
        greater.

    verbose: bool, default = True
        Logs information in console.

    Example
    -------
    >>> from transmorph.datasets import load_zhou_10x
    >>> from transmorph.models import FusedGWEmbedding
    >>> model = FusedGWEmbedding()
    >>> dataset = load_zhou_10x()
    >>> model.fit(datasets)
    """

    def __init__(
        self,
        matching_metric: str = "sqeuclidean",
        matching_metric_kwargs: Optional[Dict] = None,
        alpha: float = 0.5,
        GW_loss: Literal["square_loss", "kl_loss"] = "square_loss",
        obs_class: Optional[str] = None,
        embedding_n_neighbors: int = 10,
        embedding_optimizer: Literal["umap", "mde"] = "umap",
        pca_n_components: int = 30,
        embedding_dimension: int = 2,
        verbose: bool = True,
    ):
        from .. import settings

        if verbose:
            settings.verbose = "INFO"
        else:
            settings.verbose = "WARNING"

        # Loading algorithms
        preprocessings = [
            CommonFeatures(),
            PCA(n_components=pca_n_components, strategy="concatenate"),
        ]
        matching_alg = FusedGW(
            OT_metric=matching_metric,
            OT_metric_kwargs=matching_metric_kwargs,
            default_GW_metric=matching_metric,
            default_GW_metric_kwargs=matching_metric_kwargs,
            alpha=alpha,
            GW_loss=GW_loss,
            common_features_mode="total",
        )

        merging = GraphEmbedding(
            optimizer=embedding_optimizer,
            n_neighbors=embedding_n_neighbors,
            embedding_dimension=embedding_dimension,
            symmetrize_edges=True,
        )

        # Building model
        linput = LayerInput()
        ltransform = LayerTransformation()
        for transformation in preprocessings:
            ltransform.add_transformation(transformation=transformation)
        lmatching = LayerMatching(
            matching=matching_alg,
            obs_class=obs_class,
        )
        lmerging = LayerMerging(merging=merging)
        loutput = LayerOutput()
        linput.connect(ltransform)
        ltransform.connect(lmatching)
        lmatching.connect(lmerging)
        lmerging.connect(loutput)

        Model.__init__(self, input_layer=linput, str_identifier="EMBED_MNN")

    def transform(
        self,
        datasets: List[AnnData],
        use_representation: Optional[str] = None,
        output_representation: str = "transmorph",
    ) -> None:
        """
        Carries out the model on a list of AnnData objects. Writes the result in
        .obsm fields.

        Parameters
        ----------
        datasets: List[AnnData]
            List of anndata objects, must have at least one common feature.

        use_representation: Optional[str]
            .obsm to use as input.

        output_representation: str
            .obsm destination key, "transmorph" by default.
        """
        self.fit(
            datasets=datasets,
            reference=None,
            use_representation=use_representation,
            output_representation=output_representation,
        )
