#!/usr/bin/env python3

import anndata as ad
import numpy as np
import pymde

from pymde.preprocess import Graph
from umap.umap_ import simplicial_set_embedding, find_ab_params
from scipy.sparse import csr_matrix
from typing import List, Literal

from ..merging import Merging
from ...matching import _TypeMatchingSet
from ....utils.graph import (
    combine_matchings,
    generate_membership_matrix,
    nearest_neighbors,
)
from ....utils.matrix import extract_chunks


class GraphEmbedding(Merging):
    """
    This merging regroups different graph embedding methods (GEM). Given a
    graph of weighted edges between data samples, a GEM learns an embedding
    of samples which minimizes a cost depending on graph edges. We use as a
    representation graph a combination of matching graphs and internal kNN
    graphs. Several GEMs have been proposed, we included 2 in transmorph.

    - Uniform Manifold Approximation Projection (UMAP) [1] is a GEM which
      assumes samples to be uniformly distributed along their support
      manifold, and learns a local metric to approximate this hypothesis.
      It then simulates the graph in an embedding space, with edges
      acting as springs pulling or repelling samples depending to their
      estimated distance. We use the umap-learn implementation of UMAP.

    - Minimum Distorsion Embedding (MDE) [2] is a GEM which approaches the
      problem by trying to minimize distances between points with strong
      edges, while the embedding is constrained (in our case, it must be
      standardized). This creates an optimization problem yielding
      convincing embeddings. We ise the pymde implementation of MDE.

    Optimizers parameters can be edited in transmorph settings.

    Parameters
    ----------
    optimizer: Literal["umap", "mde"] = "umap"
        Graph embedding method to use

    n_neighbors: int = 10
        Number of neighbors to include in inner knn graphs.

    embedding_dimension: int, default = 2
        Target dimensionality of the embedding. For visualization
        purposes, choose 2 or 3. For feature learning, higher numbers
        like 20 or 50 can yield interesting results. It is recommended
        not to exceed initial dataset dimensionality.

    matching_strength: float = 10.0
        Edges weights between samples belonging to the same batch will
        be divided by this coefficient. Therefore, increasing it will
        tend to emphasize matching edges in the embedding representation.
        In the other hand, decreasing it will emphasize initial
        datasets geometry.

    include_inner_graphs: bool, default = True
        Adds edges of the kNN graph of each dataset in the final graph
        to embed. If false, only matching edges are embedded.

    symmetrize_edges: bool, default = True
        Symmetrize the graph to embed meaning if i matches j then j
        matches i. Recommended for stability, though less relevant
        with a high number of edges of good quality.

    References
    ----------
    [1] Becht, Etienne, et al. "Dimensionality reduction for visualizing
        single-cell data using UMAP." Nature biotechnology 37.1 (2019): 38-44.

    [2] Agrawal, Akshay, Alnur Ali, and Stephen Boyd. "Minimum-distortion
        embedding." arXiv preprint arXiv:2103.02559 (2021).
    """

    def __init__(
        self,
        optimizer: Literal["umap", "mde"] = "umap",
        n_neighbors: int = 10,
        embedding_dimension: int = 2,
        symmetrize_edges: bool = True,
    ):
        Merging.__init__(
            self,
            preserves_space=False,
            str_identifier="GRAPH_EMBEDDING",
            matching_mode="bool",
        )
        assert optimizer in ("umap", "mde"), f"Unknown optimizer {optimizer}."
        self.optimizer = optimizer
        self.n_neighbors = n_neighbors
        self.embedding_dimension = embedding_dimension
        self.symmetrize_edges = symmetrize_edges

    def transform(
        self, datasets: List[ad.AnnData], embeddings: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Builds a joint graph of datasets, then run the optimizer.
        """
        from .... import settings

        ndatasets = len(embeddings)

        # Matching matrix is already row-normalized
        matchings: _TypeMatchingSet = {}
        for i in range(ndatasets):
            for j in range(ndatasets):
                if i == j:
                    continue
                G = generate_membership_matrix(
                    self.get_matching(i, j),
                    embeddings[i],
                    embeddings[j],
                )
                matchings[i, j] = G
                self.log(f"Matching graph {i, j}: {(G > 0).sum()} edges.")
                self.log(
                    f"min: {G.data.min()}, max: {G.data.max()}, mean: {G.data.mean()}"
                )

        # We retrieve and scale boolean kNN-graphs
        inner_graphs = [
            nearest_neighbors(adata=adata, mode="edges", n_neighbors=self.n_neighbors)
            for adata in datasets
        ]
        for i, G in enumerate(inner_graphs):
            assert isinstance(G, csr_matrix)
            G = generate_membership_matrix(
                G,
                embeddings[i],
                embeddings[i],
            )
            inner_graphs[i] = G
            self.log(f"Internal graph {i}: {(G > 0).sum()} edges.")
            self.log(f"min: {G.data.min()}, max: {G.data.max()}, mean: {G.data.mean()}")

        # Combining all those in a big edges matrix
        edges = combine_matchings(
            matchings=matchings,
            knn_graphs=inner_graphs,
            symmetrize=self.symmetrize_edges,
            target_edges=self.n_neighbors,
        )

        # Checking total number of edges
        n_edges = (edges > 0).sum()
        self.log(
            f"Embedding a graph of {n_edges} edges in "
            f"{self.embedding_dimension} dimensions."
        )
        f"min: {edges.data.min()}, max: {edges.data.max()}, mean: {edges.data.mean()}"
        if n_edges > settings.large_number_edges:
            self.warn(
                f"High number of edges detected ({n_edges} > "
                f"{settings.large_number_edges}). This may take some "
                "time. Using 'subsampling' option, decreasing the number "
                "of neighbors or changing Matching "
                "algorithm may accelerate the convergence."
            )

        # Computing the embedding
        if self.optimizer == "umap":
            nsamples = sum(X.shape[0] for X in embeddings)
            n_epochs = (
                settings.umap_maxiter
                if settings.umap_maxiter is not None
                else 500
                if nsamples < settings.large_dataset_threshold
                else 200
            )
            if settings.umap_a is None or settings.umap_b is None:
                a, b = find_ab_params(settings.umap_spread, settings.umap_min_dist)
            else:
                a, b = settings.umap_a, settings.umap_b
            self.log("Computing UMAP...")
            X_result, _ = simplicial_set_embedding(
                data=None,
                graph=edges,
                n_components=self.embedding_dimension,
                initial_alpha=settings.umap_alpha,
                a=a,
                b=b,
                gamma=settings.umap_gamma,
                negative_sample_rate=settings.umap_negative_sample_rate,
                n_epochs=n_epochs,
                init="spectral",
                random_state=settings.umap_random_state,
                metric=settings.umap_metric,
                metric_kwds=settings.umap_metric_kwargs,
                densmap=False,
                densmap_kwds={},
                output_dens=False,
                verbose=False,
            )
        elif self.optimizer == "mde":
            self.log("Computing MDE...")
            mde = pymde.preserve_neighbors(
                Graph(edges),
                embedding_dim=self.embedding_dimension,
                constraint=pymde.Standardized(),
                init=settings.mde_initialization,
                repulsive_fraction=settings.mde_repulsive_fraction,
                device=settings.mde_device,
            )
            X_result = np.array(mde.embed())
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}.")
        return extract_chunks(X_result, [X.shape[0] for X in embeddings])
