#!/usr/bin/env python3

from typing import List, Optional, Union
import numpy as np
import pymde

from anndata import AnnData
from pymde.preprocess import Graph
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix

from .mergingABC import MergingABC
from ..matching.matchingABC import MatchingABC
from ..utils import nearest_neighbors
from ..utils.anndata_interface import get_matrix


def combine_matchings(
    datasets: List[AnnData],
    knn_graphs: List[csr_matrix],
    matching: Optional[MatchingABC] = None,
    matching_mtx: Optional[Union[csr_matrix, np.ndarray]] = None,
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
    for i, adata_i in enumerate(datasets):
        # Initial relations
        knn_graph = knn_graphs[i].tocoo()
        rows += list(knn_graph.row + offset_i)
        cols += list(knn_graph.col + offset_i)
        data += list(knn_graph.data)
        # Matchings
        offset_j = 0
        ni = datasets[i].X.shape[0]
        for j, adata_j in enumerate(datasets):
            nj = datasets[j].X.shape[0]
            if i >= j:
                offset_j += nj
                continue
            if matching is not None:
                T = matching.get_matching(adata_i, adata_j)
            elif matching_mtx is not None:  # Works as an elif
                T = matching_mtx
                if T.shape[0] == adata_j.n_obs:
                    T = T.T
                if type(T) is csc_matrix or type(T) is csr_matrix:
                    T = T.toarray()
                assert type(T) is np.ndarray, f"Unrecognized type: {type(T)}"
                T = csr_matrix(T / T.sum(axis=1, keepdims=True))
            else:
                raise AssertionError("matching or matching_mtx must be set.")
            matching_ij = T.tocoo()
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
        embedding_dimension: int = 2,
        initialization: str = "quadratic",
        n_neighbors: int = 10,
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
        for dataset in datasets:
            X = get_matrix(dataset, X_kw)
            inner_graphs.append(
                nearest_neighbors(
                    X,
                    n_neighbors=self.n_neighbors,
                    metric=self.knn_metric,
                    metric_kwargs=self.knn_metric_kwargs,
                    include_self_loops=False,
                )
            )
        edges = combine_matchings(datasets, inner_graphs, matching, matching_mtx)
        edges.data = np.clip(edges.data, 0.0, 1.0)
        edges = edges + edges.T - edges.multiply(edges.T)  # symmetrize
        edges.data[edges.data == 0] = 1e-9
        # Gaussian model
        # pij = exp(-dij**2 * lambda)
        # iff dij = sqrt(-ln(pij) / lambda)
        # + epsilon to stabilize MDE solver
        edges.data = np.sqrt(-np.log(edges.data) / self.concentration) + 1e-9
        mde = pymde.preserve_neighbors(
            Graph(edges),
            embedding_dim=self.embedding_dimension,
            constraint=pymde.Standardized(),
            init=self.initialization,
            repulsive_fraction=self.repulsive_fraction,
            device=self.device,
            verbose=self.verbose,
        )
        X_tot = mde.embed(verbose=self.verbose)
        result = []
        offset = 0
        for adata in datasets:
            n_obs = adata.n_obs
            result.append(X_tot[offset : offset + n_obs].numpy())
            offset += n_obs
        return result
