#!/usr/bin/env python3

from anndata import AnnData
from transmorph.engine import Model
from transmorph.engine.layers import (
    LayerInput,
    LayerTransformation,
    LayerMatching,
    LayerMerging,
    LayerOutput,
)
from transmorph.engine.matching import MNN
from transmorph.engine.merging import GraphEmbedding
from transmorph.engine.subsampling import VertexCover
from transmorph.engine.transforming import CommonFeatures, Standardize, PCA
from typing import Dict, List, Literal, Optional


class EmbedMNN:
    """
    This model performs preprocessing steps, then carries out mutual nearest
    neighbors (MNN) between pairs of datasets. It eventually embeds a combination
    of datasets kNN graphs and pairwise MNN graphs into a low dimensional space,
    in which further analyzes such as clustering can be carried out.

    Parameters
    ----------
    mnn_n_neighbors: int, default = 30
        Number of neighbors to use for the mutual nearest neighbors step.

    mnn_metric: str, default = "sqeuclidean"
        Metric to use during MNN step.

    mnn_kwargs: Optional[Dict], default = None
        Additional metric parameters.

    inner_n_neighbors: int, default = 10
        Number of neighbors to use in the kNN step.

    embedding_optimizer: Literal["umap", "mde"], default = "umap"
        Graph embedding algorithm to use.

    embedding_dimension: int, default = 2
        Number of dimensions in the final embedding.

    matching_strength: float, default = 1.0
        Increase this value to tune MNN edges strength during graph
        embedding algorithm.

    n_components: int, default = 30
        Number of principal components to use if data dimensionality is
        greater.
    """

    def __init__(
        self,
        mnn_n_neighbors: int = 30,
        mnn_metric: str = "sqeuclidean",
        mnn_kwargs: Optional[Dict] = None,
        inner_n_neighbors: int = 10,
        embedding_optimizer: Literal["umap", "mde"] = "umap",
        embedding_dimension: int = 2,
        matching_strength: float = 1.0,
        n_components: int = 30,
        use_subsampling: bool = False,
        verbose: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    ):
        self.n_components = n_components
        self.verbose = verbose

        # Loading algorithms
        preprocessings = [
            CommonFeatures(),
            Standardize(center=True, scale=True),
            PCA(n_components=n_components, strategy="concatenate"),
        ]
        subsampling = None
        if use_subsampling:
            self.subsampling = VertexCover(n_neighbors=3)
            mnn_n_neighbors /= 3
        matching = MNN(
            metric=mnn_metric,
            metric_kwargs=mnn_kwargs,
            n_neighbors=mnn_n_neighbors,
            common_features_mode="total",
            solver="auto",
        )
        merging = GraphEmbedding(
            optimizer=embedding_optimizer,
            n_neighbors=inner_n_neighbors,
            embedding_dimension=embedding_dimension,
            matching_strength=matching_strength,
        )

        # Building model
        linput = LayerInput()
        ltransform = LayerTransformation()
        for transformation in preprocessings:
            ltransform.add_transformation(transformation=transformation)
        lmatching = LayerMatching(matching=matching, subsampling=subsampling)
        lmerging = LayerMerging(merging=merging)
        loutput = LayerOutput()
        linput.connect(ltransform)
        ltransform.connect(lmatching)
        lmatching.connect(lmerging)
        lmerging.connect(loutput)
        self.model = Model(input_layer=linput)

    def transform(
        self,
        datasets: List[AnnData],
        use_representation: Optional[str] = None,
    ) -> None:
        """
        Carries out the model on a list of AnnData objects. Writes the result in
        .obsm fields.

        Parameters
        ----------
        datasets: List[AnnData]
            List of anndata objects, must have at least one common feature.
        """
        self.model.fit(
            datasets=datasets,
            reference=None,
            use_representation=use_representation,
        )
