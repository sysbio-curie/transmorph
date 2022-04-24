#!/usr/bin/env python3

import numpy as np
import pymde

from pymde.preprocess import Graph
from umap.umap_ import simplicial_set_embedding, find_ab_params
from typing import List, Literal

from ..merging import Merging
from ...matching import _TypeMatchingSet
from ...traits.usesneighbors import UsesNeighbors
from ....utils.graph import combine_matchings
from ....utils.matrix import extract_chunks


class GraphEmbedding(Merging, UsesNeighbors):
    """
    TODO update this text to account for UMAP

    Minimum Distorsion Integration is a joint embedding technique
    expressed in the Minimum Distorsion Embedding framework [1]. Given
    n datasets linked together by {2 choose n} fuzzy matchings, it
    computes a joint embedding of both datasets so that matched samples
    are brought close from one another. Furthermore, initial graph
    structure of each dataset is included in the problem, so that close
    points of a dataset tend to end up close from one another in the
    final embedding.

    We rely on the MDE implementation of pyMDE that is fast and scalable.

    Parameters
    ----------
    matching: MatchingABC
        Fitted, referenced matching between datasets.

    embedding_dimension: int, default = 2
        Target dimensionality of the embedding. For visualization
        purposes, choose 2 or 3. For feature learning, higher numbers
        like 20 or 50 can yield interesting results. It is recommended
        not to exceed initial dataset dimensionality.

    initialization: str, default = "quadratic"
        Initialization strategy, "quadratic" or "random".

    n_neighbors: int, default = 10
        Number of neighbors to include in inner knn graphs.

    knn_metric: str, default = "sqeuclidean"
        Metric used for computing knn graph.

    knn_metric_kwargs: dict = {}
        Additional metric arguments for scipy.cdist.

    repulsive_fraction: float, default = 0.5
        How many repulsive edges to include, relative to the number
        of attractive edges. 1.0 means as many repulsive edges as attractive
        edges. The higher this number, the more uniformly spread out the
        embedding will be. Defaults to 0.5 for standardized embeddings, and
        1 otherwise.

    device: str, default = "cpu"
        Device for the embedding (eg, 'cpu', 'cuda').

    verbose: bool
        If ``True``, print verbose output.

    References
    ----------
    [1] A. Agrawal, A. Ali, S. Boyd, Minimum-Distorsion Embedding, 2021
    """

    def __init__(
        self,
        optimizer: Literal["umap", "mde"] = "umap",
        embedding_dimension: int = 2,
        mass_concentration: float = 1.0,
    ):
        Merging.__init__(
            self,
            preserves_space=False,
            str_identifier="GRAPH_EMBEDDING",
            matching_mode="normalized",
        )
        UsesNeighbors.__init__(self)
        assert optimizer in ("umap", "mde"), f"Unknown optimizer {optimizer}."
        self.optimizer = optimizer
        self.embedding_dimension = embedding_dimension
        self.concentration = mass_concentration

    def transform(self, datasets: List[np.ndarray]) -> List[np.ndarray]:
        """
        Builds a joint graph of datasets, then run the optimizer.
        """
        from .... import settings

        ndatasets = len(datasets)
        inner_graphs = [
            self.get_neighbors_graph(i, mode="distances") for i in range(ndatasets)
        ]
        matchings: _TypeMatchingSet = {}
        for i in range(ndatasets):
            for j in range(i + 1, ndatasets):
                matchings[i, j] = self.get_matching(i, j)
                matchings[j, i] = self.get_matching(j, i)
        edges = combine_matchings(
            inner_graphs, matchings, "distance", self.concentration
        )
        if self.optimizer == "umap":
            nsamples = sum(X.shape[0] for X in datasets)
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
        return extract_chunks(X_result, [X.shape[0] for X in datasets])
