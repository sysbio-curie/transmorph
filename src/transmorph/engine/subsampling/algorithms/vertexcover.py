#!/usr/bin/env python3

import numpy as np

from typing import List, Tuple

from ..subsampling import Subsampling
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
    """

    def __init__(
        self,
        n_hops: int = 1,
    ):
        Subsampling.__init__(self, str_identifier="VERTEX_COVER")
        UsesNeighbors.__init__(self)
        self.n_hops = n_hops

    def subsample(
        self, datasets: List[np.ndarray]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Simply retrieves neighbor graphs, and uses it to compute vertex covers.
        """
        results = []
        for i, _ in enumerate(datasets):
            Adj = UsesNeighbors.get_neighbors_graph(i, mode="edges")
            anchors, references = vertex_cover(Adj, hops=self.n_hops)
            results.append((anchors, references))
        return results
