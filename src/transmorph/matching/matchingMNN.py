#!/usr/bin/env python3

from typing import Union, Callable

import copy
import numpy as np
from scipy.sparse.coo import coo_matrix

import scanpy as sc

from .matchingABC import MatchingABC
from ..TData import TData
from ..utils import nearest_neighbors, pca_multi, vertex_cover


class MatchingMNN(MatchingABC):
    """
    Mutual Nearest Neighbors (MNN) matching. Two samples xi and yj
    are matched if xi belongs to the k-nearest neighbors (kNNs) of yj
    and vice-versa. If we denote by dk(x) the distance from x to its
    kNN, then xi and yj are matched if d(xi, yj) < min{dk(xi), dk(yj)}.

    Parameters
    ----------
    metric: str or Callable, default = "sqeuclidean"
        Scipy-compatible metric.

    metric_kwargs: dict, default = {}
        Additional metric parameters.

    n_neighbors: int, default = 10
        Number of neighbors to build the MNN graph.

    n_pcs: int, default = -1
        Number of PCs to use in the PCA representation. If -1, do
        not use a PCA representation.

    use_common_features: bool, default = False
        Use pairwise common features subspace between all pairs of
        datasets. Necessitates each TData.metadata to contain a
        "features" key, a list of features names.

    use_sparse: bool, default = True
        Save matching as sparse matrices.
    """

    def __init__(
        self,
        metric: Union[str, Callable] = "sqeuclidean",
        metric_kwargs: dict = {},
        n_neighbors: int = 10,
        n_pcs: int = -1,
        use_common_features: bool = False,
        use_sparse: bool = True,
        use_vertex_cover: bool = False,
        verbose: bool = False,
    ):
        metadata = ["features"] if use_common_features else []
        MatchingABC.__init__(self, use_sparse=use_sparse, metadata_needed=metadata)
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.n_neighbors = n_neighbors
        if n_pcs != -1 and n_pcs <= 0:
            raise ValueError("n_pcs <= 0")
        self.n_pcs = n_pcs
        self.use_common_features = use_common_features
        self.use_vertex_cover = use_vertex_cover
        self.verbose = verbose

    def _check_input(self, adata: sc.AnnData):
        if not MatchingABC._check_input(self, adata):
            return False
        if self.n_pcs >= 1 and adata.X.shape[1] < self.n_pcs:
            print("n_pcs >= X.shape[1]")
            return False
        if self.verbose:
            print(f"Successfully checked {adata}.")
        return True

    def _preprocess(self, adata1: TData, adata2: TData):
        preprocess_needed = (
            self.n_pcs >= 1 or self.use_common_features or self.use_vertex_cover
        )
        if not preprocess_needed:
            if self.verbose:
                print(f"No need to preprocess {adata1}, {adata2}.")
            return adata1, adata2
        if self.verbose:
            print(f"Preprocessing {adata1} and {adata2}).")
        adata1_new, adata2_new = copy.deepcopy(adata1), copy.deepcopy(adata2)
        if self.use_common_features:
            f1 = adata1.uns["_transmoph"]["matching"]["features"]
            f2 = adata2.uns["_transmoph"]["matching"]["features"]
            fcommon = np.intersect1d(f1, f2)
            f1idx = np.argsort(f1)
            f2idx = np.argsort(f2)
            sel1 = np.isin(f1[f1idx], fcommon)
            sel2 = np.isin(f2[f2idx], fcommon)
            adata1_new.X = adata1_new.X[:, f1idx[sel1]]
            adata1_new.uns["_transmoph"]["matching"]["features"] = f1[f1idx][sel1]
            adata2_new.X = adata2_new.X[:, f2idx[sel2]]
            adata2_new.uns["_transmoph"]["matching"]["features"] = f2[f2idx][sel2]
            if self.verbose:
                print(f"Selected {len(fcommon)} common features.")
        if self.n_pcs >= 1:
            if self.verbose:
                print("Computing common PCA...")
            adata1_new.X, adata2_new.X = pca_multi(
                [adata1_new.X, adata2_new.X], n_components=self.n_pcs
            )
        if self.use_vertex_cover:
            for adata in (adata1_new, adata2_new):
                if "anchors" in adata.uns["_transmoph"]["matching"]:
                    continue
                if self.verbose:
                    print(f"Computing vertex cover of {adata}...")
                use_nndescent = adata1.X.shape[0] > 4096  # NNDescent if large dataset
                A = nearest_neighbors(
                    adata.X,
                    n_neighbors=self.n_neighbors,
                    metric=self.metric,
                    metric_kwargs=self.metric_kwargs,
                    use_nndescent=use_nndescent,
                )
                anchors, mapping = vertex_cover(A, hops=1)
                adata.uns["_transmoph"]["matching"]["anchors"] = anchors
                adata.uns["_transmoph"]["matching"]["mapping"] = mapping
        return adata1_new, adata2_new

    def _match2(self, adata1: sc.AnnData, adata2: sc.AnnData):
        if self.verbose:
            print(f"Matching {adata1} against {adata2}.")
        X, Y = adata1.X, adata2.X
        if self.use_vertex_cover:
            X = X[adata1.metadata["anchors"]]
            Y = Y[adata2.metadata["anchors"]]
        T = nearest_neighbors(
            X,
            Y=Y,
            metric=self.metric,
            metric_kwargs=self.metric_kwargs,
            n_neighbors=self.n_neighbors,
            use_nndescent=False,
        )
        if self.use_vertex_cover:
            cov_to_full_1 = np.arange(adata1.shape[0])[adata1.metadata["anchors"]]
            cov_to_full_2 = np.arange(adata2.shape[0])[adata2.metadata["anchors"]]
            T = T.tocoo()
            rows, cols, data = [], [], []
            for i, j in zip(T.row, T.col):
                rows.append(cov_to_full_1[i])
                cols.append(cov_to_full_2[j])
                data.append(1)
            T = coo_matrix(
                (data, (rows, cols)), shape=(adata1.shape[0], adata2.shape[0])
            ).tocsr()
        return T
