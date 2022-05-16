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
from transmorph.engine.matching import MNN
from transmorph.engine.merging import GraphEmbedding
from transmorph.engine.subsampling import VertexCover
from transmorph.engine.transforming import CommonFeatures, PCA


class EmbedMNN(Model):
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

    use_subsampling: bool, default = False
        Run MNN and LISI on a subsample of points to spare performance.
        Useful for large datasets.

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

    include_inner_graphs: bool, default = True
        Adds edges of the kNN graph of each dataset in the final graph
        to embed. If false, only matching edges are embedded.

    symmetrize_edges: bool, default = True
        Symmetrize the graph to embed meaning if i matches j then j
        matches i. Recommended for stability, though less relevant
        with a high number of edges of good quality.


    verbose: bool, default = True
        Logs runtime information in console.
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
        include_inner_graphs: bool = True,
        symmetrize_edges: bool = True,
        use_subsampling: bool = False,
        verbose: bool = True,
    ):
        from .. import settings

        if verbose:
            settings.verbose = "INFO"
        else:
            settings.verbose = "WARNING"

        self.n_components = n_components

        # Loading algorithms
        preprocessings = [
            CommonFeatures(),
            PCA(n_components=n_components, strategy="concatenate"),
        ]
        subsampling = None
        if use_subsampling:
            self.subsampling = VertexCover()
            mnn_n_neighbors = max(5, int(mnn_n_neighbors / 3))
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
            include_inner_graphs=include_inner_graphs,
            symmetrize_edges=symmetrize_edges,
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

        Model.__init__(self, input_layer=linput, str_identifier="EMBED_MNN")

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

        use_representation: Optional[str]
            .obsm to use as input.
        """
        self.fit(
            datasets=datasets,
            reference=None,
            use_representation=use_representation,
        )
