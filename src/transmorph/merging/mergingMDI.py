#!/usr/bin/env python3

from logging import warn
from typing import List, Optional, Union
import numpy as np
import pymde

from anndata import AnnData
from pymde.preprocess import Graph
from scipy.sparse import csr_matrix

from .mergingABC import MergingABC
from ..matching.matchingABC import MatchingABC
from ..utils import nearest_neighbors, extract_chunks
from ..utils.anndata_interface import get_matrix
from ..utils.graph import combine_matchings


class MergingMDI(MergingABC):
    """
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
        embedding_dimension: int = 2,
        initialization: str = "quadratic",
        n_neighbors: int = 15,
        knn_metric: str = "sqeuclidean",
        knn_metric_kwargs: dict = {},
        repulsive_fraction: float = 1.0,
        concentration: float = 1.0,
        device: str = "cpu",
        verbose: bool = False,
    ):
        super().__init__(use_reference=False)
        self.embedding_dimension = embedding_dimension
        self.initialization = initialization
        self.n_neighbors = n_neighbors
        self.knn_metric = knn_metric
        self.knn_metric_kwargs = knn_metric_kwargs
        self.repulsive_fraction = repulsive_fraction
        self.concentration = concentration
        self.device = device
        self.verbose = verbose

    def fit(
        self,
        datasets: List[AnnData],
        matching: Optional[MatchingABC] = None,
        matching_mtx: Optional[Union[csr_matrix, np.ndarray]] = None,
        X_kw: str = "",
        reference_idx: int = -1,
    ) -> List[np.ndarray]:
        self._check_input(datasets, matching, matching_mtx, X_kw, reference_idx)
        inner_graphs = []
        is_high_dim = False
        for dataset in datasets:
            X = get_matrix(dataset, X_kw)
            if X.shape[1] > 100:
                is_high_dim = True
            inner_graphs.append(
                nearest_neighbors(
                    X,
                    n_neighbors=self.n_neighbors,
                    metric=self.knn_metric,
                    metric_kwargs=self.knn_metric_kwargs,
                    include_self_loops=False,
                )
            )
        if is_high_dim:
            warn(
                "High dimensional data detected (D>100)."
                "You may want to reduce dimensionality first."
            )
        ndatasets = len(datasets)
        matchings = {}
        if matching is not None:
            for i in range(ndatasets):
                for j in range(i):
                    T = matching.get_matching(datasets[i], datasets[j])
                    matchings[i, j] = T
                    matchings[j, i] = T
        elif matching_mtx is not None:
            matchings[0, 1] = matching_mtx
            matchings[1, 0] = matching_mtx

        edges = combine_matchings(
            inner_graphs, matchings, "distance", self.concentration
        )
        mde = pymde.preserve_neighbors(
            Graph(edges),
            embedding_dim=self.embedding_dimension,
            constraint=pymde.Standardized(),
            init=self.initialization,
            repulsive_fraction=self.repulsive_fraction,
            device=self.device,
            verbose=self.verbose,
        )
        X_result = np.array(mde.embed(verbose=self.verbose))
        return extract_chunks(X_result, [adata.n_obs for adata in datasets])
