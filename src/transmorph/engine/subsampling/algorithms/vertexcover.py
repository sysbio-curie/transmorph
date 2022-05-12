#!/usr/bin/env python3

import numpy as np

from typing import List, Optional

from ..subsampling import Subsampling, _TypeSubsampling
from ...traits.usesneighbors import UsesNeighbors
from ....utils.graph import vertex_cover


class VertexCover(Subsampling, UsesNeighbors):
    """
    Subsamples the dataset by computing a vertex cover, where every point
    in the dataset either belongs to the subsampling or is neighbor to a
    subsampled point (in the sense of neighborhood graph). We use a heuristic
    in O(n) that returns in worst case scenario a vertex cover twice as big
    as the smallest one.

    Parameters
    ----------
    n_hops: int, default = 1
        Maximal geodesic distance a point is allowed to be to the cover.

    n_neighbors: int, default = 5
        Number of neighbors to use to build the kNN-graph along which is
        computed the vertex cover. On average, a dataset with n samples
        will have a subsampled size of n / n_neighbors.
    """

    def __init__(self, n_hops: int = 1, n_neighbors: Optional[int] = None):
        from .... import settings, use_setting

        Subsampling.__init__(self, str_identifier="VERTEX_COVER")
        UsesNeighbors.__init__(self)
        self.n_hops = n_hops
        self.n_neighbors = use_setting(n_neighbors, settings.vertexcover_n_neighbors)

    def subsample(self, datasets: List[np.ndarray]) -> List[_TypeSubsampling]:
        """
        Simply retrieves neighbor graphs, and uses it to compute vertex covers.
        """
        results = []
        for i, _ in enumerate(datasets):
            Adj = UsesNeighbors.get_neighbors_graph(
                i, mode="edges", n_neighbors=self.n_neighbors
            )
            anchors, references = vertex_cover(Adj, hops=self.n_hops)
            results.append((anchors, references))
        return results
