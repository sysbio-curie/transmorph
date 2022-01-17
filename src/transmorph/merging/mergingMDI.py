#!/usr/bin/env python3

from typing import List
import numpy as np
import pymde

from pymde.preprocess import Graph
from scipy.sparse import csr_matrix, coo_matrix

from .mergingABC import MergingABC
from ..matching.matchingABC import MatchingABC
from ..utils import nearest_neighbors


def combine_matchings(
    matching: MatchingABC,
    knn_graphs: List[csr_matrix],
) -> csr_matrix:
    """
    Concatenates any number of matchings Mij and knn-graph
    adjacency matrices Ai in a single sparse matrix T. Diagonal
    blocks of T is composed by Ai matrices, and ij block is
    Mij if i < j otherwise Mji.T

    matching: MatchingABC
        A fitted MatchingABC object with no reference.

    knn_graph: List[csr_matrix]
        List of knn-graphs, where knn-graph[i] is the knn-graph
        associated to matching.datasets[i].
    """
    rows, cols, data, N = [], [], [], 0
    offset_i = 0
    for i in range(matching.n_datasets):
        # Initial relations
        knn_graph = knn_graphs[i].tocoo()
        rows += list(knn_graph.row + offset_i)
        cols += list(knn_graph.col + offset_i)
        data += list(knn_graph.data)
        # Matchings
        offset_j = 0
        ni = matching.datasets[i].shape[0]
        for j in range(matching.n_datasets):
            nj = matching.datasets[j].shape[0]
            if i >= j:
                offset_j += nj
                continue
            matching_ij = matching.get_matching(i, j, normalize=True)
            if type(matching_ij) is np.ndarray:
                matching_ij = coo_matrix(matching_ij)
            elif type(matching_ij) is csr_matrix:
                matching_ij = matching_ij.tocoo()
            else:
                raise NotImplementedError
            rows_k, cols_k = matching_ij.row, matching_ij.col
            rows_k += offset_i
            cols_k += offset_j
            rows += list(rows_k)  # Keep the symmetry
            rows += list(cols_k)
            cols += list(cols_k)
            cols += list(rows_k)
            data += 2 * len(cols_k) * [1]
            offset_j += nj
        offset_i += ni
        N += ni
    return coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()


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
        matching: MatchingABC,
        embedding_dimension: int = 2,
        initialization: str = "quadratic",
        n_neighbors: int = 10,
        knn_metric: str = "sqeuclidean",
        knn_metric_kwargs: dict = {},
        repulsive_fraction: float = 1.0,
        device: str = "cpu",
        verbose: bool = False,
    ):
        MergingABC.__init__(self, matching)
        self.embedding_dimension = embedding_dimension
        self.initialization = initialization
        self.n_neighbors = n_neighbors
        self.knn_metric = knn_metric
        self.knn_metric_kwargs = knn_metric_kwargs
        self.repulsive_fraction = repulsive_fraction
        self.device = device
        self.verbose = verbose

    def _check_input(self) -> None:
        """
        Raises an additional warning if some source samples are unmatched.
        """
        super()._check_input()

    def transform(self) -> np.ndarray:
        inner_graphs = []
        for dataset in self.matching.datasets:
            inner_graphs.append(
                nearest_neighbors(
                    dataset,
                    n_neighbors=self.n_neighbors,
                    metric=self.knn_metric,
                    metric_kwargs=self.knn_metric_kwargs,
                    include_self_loops=False,
                    use_nndescent=True,
                )
            )
        edges = combine_matchings(self.matching, inner_graphs)
        edges.data = (
            1 - edges.data + 1e-8
        )  # TODO: improve conversion weight -> distance
        mde = pymde.preserve_neighbors(
            Graph(edges),
            embedding_dim=self.embedding_dimension,
            constraint=pymde.Standardized(),
            init=self.initialization,
            repulsive_fraction=self.repulsive_fraction,
            device=self.device,
            verbose=self.verbose,
        )
        return mde.embed(verbose=self.verbose)
