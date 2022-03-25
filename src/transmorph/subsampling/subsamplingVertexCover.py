#!/usr/bin/env python3

import numpy as np

from anndata import AnnData
from typing import Dict

from transmorph.utils.anndata_interface import get_matrix
from .subsamplingABC import SubsamplingABC

from ..utils.graph import nearest_neighbors, vertex_cover


class SubsamplingVertexCover(SubsamplingABC):
    """
    Subsamples the dataset by computing a vertex cover, where every point
    in the dataset either belongs to the subsampling or is neighbor to a
    subsampled point (in the sense of neighborhood graph). We use a heuristic
    in O(n) that returns in worst case scenario a vertex cover twice as big
    as the smallest one.

    Parameters
    ----------
    n_neighbors: int, default = 5
        Number of neighbors to use when building the n eighborhood graph.

    n_hops: int, default = 1
        Maximal geodesic distance a point is allowed to be to the cover.

    metric: str, default = "sqeuclidean"
        Scipy metric to use when building the neighborhood graph.

    metric_kwargs: dict, default = {}
        Additional metric parameters
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        n_hops: int = 1,
        metric: str = "sqeuclidean",
        metric_kwargs: Dict = {},
    ):
        self.n_neighbors = n_neighbors
        self.n_hops = n_hops
        self.metric = metric
        self.metric_kwargs = metric_kwargs

    def _subsample_one(self, adata: AnnData, X_kw: str = "") -> Dict[str, np.ndarray]:
        X = get_matrix(adata, X_kw)
        use_nndescent = adata.n_obs > 2048
        Adj = nearest_neighbors(
            X,
            n_neighbors=self.n_neighbors,
            use_nndescent=use_nndescent,
            metric=self.metric,
            metric_kwargs=self.metric_kwargs,
        )
        anchors, references = vertex_cover(Adj, hops=self.n_hops)
        return {"is_anchor": anchors, "ref_anchor": references}
