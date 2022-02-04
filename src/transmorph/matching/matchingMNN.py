#!/usr/bin/env python3

from typing import Union, Callable

import copy
import numpy as np
from scipy.sparse.coo import coo_matrix

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

    def _check_input(self, t: TData):
        if not MatchingABC._check_input(self, t):
            return False
        if self.n_pcs >= 1 and t.X.shape[1] < self.n_pcs:
            print("n_pcs >= X.shape[1]")
            return False
        if self.verbose:
            print(f"Successfully checked {str(t)}.")
        return True

    def _preprocess(self, t1: TData, t2: TData):
        preprocess_needed = (
            self.n_pcs >= 1 or self.use_common_features or self.use_vertex_cover
        )
        if not preprocess_needed:
            if self.verbose:
                print(f"No need to preprocess {str(t1)}, {str(t2)}.")
            return t1, t2
        if self.verbose:
            print(f"Preprocessing {str(t1)} and {str(t2)}).")
        t1_new, t2_new = copy.deepcopy(t1), copy.deepcopy(t2)
        if self.use_common_features:
            f1 = t1.metadata["features"]
            f2 = t2.metadata["features"]
            fcommon = np.intersect1d(f1, f2)
            f1idx = np.argsort(f1)
            f2idx = np.argsort(f2)
            sel1 = np.isin(f1[f1idx], fcommon)
            sel2 = np.isin(f2[f2idx], fcommon)
            t1_new.X = t1_new.X[:, f1idx[sel1]]
            t1_new.metadata["features"] = f1[f1idx][sel1]
            t2_new.X = t2_new.X[:, f2idx[sel2]]
            t2_new.metadata["features"] = f2[f2idx][sel2]
            if self.verbose:
                print(f"Selected {len(fcommon)} common features.")
        if self.n_pcs >= 1:
            if self.verbose:
                print("Computing common PCA...")
            t1_new.X, t2_new.X = pca_multi(
                [t1_new.X, t2_new.X], n_components=self.n_pcs
            )
        if self.use_vertex_cover:
            for t in (t1_new, t2_new):
                if "anchors" in t.metadata:
                    continue
                if self.verbose:
                    print(f"Computing vertex cover of {str(t)}...")
                use_nndescent = t1.X.shape[0] > 4096  # NNDescent if large dataset
                A = nearest_neighbors(
                    t.X,
                    n_neighbors=self.n_neighbors,
                    metric=self.metric,
                    metric_kwargs=self.metric_kwargs,
                    use_nndescent=use_nndescent,
                )
                anchors, mapping = vertex_cover(A, hops=1)
                t.metadata["anchors"] = anchors
                t.metadata["mapping"] = mapping
        return t1_new, t2_new

    def _match2(self, t1: TData, t2: TData):
        if self.verbose:
            print(f"Matching {str(t1)} against {str(t2)}.")
        X, Y = t1.X, t2.X
        if self.use_vertex_cover:
            X = X[t1.metadata["anchors"]]
            Y = Y[t2.metadata["anchors"]]
        T = nearest_neighbors(
            X,
            Y=Y,
            metric=self.metric,
            metric_kwargs=self.metric_kwargs,
            n_neighbors=self.n_neighbors,
            use_nndescent=False,
        )
        if self.use_vertex_cover:
            cov_to_full_1 = np.arange(t1.shape[0])[t1.metadata["anchors"]]
            cov_to_full_2 = np.arange(t2.shape[0])[t2.metadata["anchors"]]
            T = T.tocoo()
            rows, cols, data = [], [], []
            for i, j in zip(T.row, T.col):
                rows.append(cov_to_full_1[i])
                cols.append(cov_to_full_2[j])
                data.append(1)
            T = coo_matrix(
                (data, (rows, cols)), shape=(t1.shape[0], t2.shape[0])
            ).tocsr()
        return T
