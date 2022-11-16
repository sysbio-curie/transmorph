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
from transmorph.engine.merging import GraphEmbedding
from transmorph.engine.transforming import CommonFeatures, PCA


class EmbedMNN(Model):
    """
    This model performs preprocessing steps, then carries out mutual nearest
    neighbors (MNN) between pairs of datasets. It eventually embeds a combination
    of datasets kNN graphs and pairwise MNN graphs into a low dimensional space,
    in which further analyzes such as clustering can be carried out.

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
    >>> from transmorph.models import EmbedMNN
    >>> model = EmbedMNN()
    >>> dataset = load_zhou_10x()
    >>> model.fit(datasets)
    """

    def __init__(
        self,
        matching: Literal["mnn", "bknn"] = "bknn",
        matching_n_neighbors: Optional[int] = None,
        matching_metric: str = "sqeuclidean",
        matching_metric_kwargs: Optional[Dict] = None,
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
            )
        else:
            raise ValueError(
                f"Unrecognized matching: {matching}. Expected 'mnn' or 'bknn'."
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
        output_representation: Optional[str] = None,
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
