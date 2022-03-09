#!/usr/bin/env python3

from typing import Tuple, Union, Callable

import copy
import numpy as np

from anndata import AnnData

from .matchingABC import MatchingABC
from ..utils import nearest_neighbors


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
        use_common_features: bool = False,
        use_vertex_cover: bool = False,
        verbose: bool = False,
    ):
        super().__init__(metadata_keys=[])
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.n_neighbors = n_neighbors
        self.use_common_features = use_common_features
        self.verbose = verbose

    def _check_input(self, adata: AnnData, dataset_key: str = "") -> None:
        """
        TODO
        """
        pass

    def _preprocess(
        self, adata1: AnnData, adata2: AnnData, dataset_key: str
    ) -> Tuple[AnnData, AnnData]:
        if self.verbose:
            print(f"Preprocessing {adata1} and {adata2}).")
        adata1_new, adata2_new = copy.deepcopy(adata1), copy.deepcopy(adata2)
        if self.use_common_features:
            f1 = adata1.var_names
            f2 = adata2.var_names
            fcommon = np.intersect1d(f1, f2)
            f1idx = np.argsort(f1)
            f2idx = np.argsort(f2)
            sel1 = np.isin(f1[f1idx], fcommon)
            sel2 = np.isin(f2[f2idx], fcommon)
            adata1_new = adata1[:, sel1]
            adata2_new = adata2[:, sel2]
            if self.verbose:
                print(f"Selected {len(fcommon)} common features.")
        return adata1_new, adata2_new

    def _match2(self, adata1: AnnData, adata2: AnnData):
        if self.verbose:
            print(f"Matching {adata1} against {adata2}.")
        X, Y = adata1.X, adata2.X
        T = nearest_neighbors(
            X,
            Y=Y,
            metric=self.metric,
            metric_kwargs=self.metric_kwargs,
            n_neighbors=self.n_neighbors,
            use_nndescent=False,
        )
        return T
