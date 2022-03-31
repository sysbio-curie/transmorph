#!/usr/bin/env python3

from .dimred import pca, pca_multi
from .geometry import sparse_cdist
from .graph import nearest_neighbors, vertex_cover
from .plotting import plot_result
from .stats import (
    earth_movers_distance,
    matching_divergence,
    neighborhood_preservation,
)

__all__ = [
    "pca",
    "pca_multi",
    "plot_result",
    "nearest_neighbors",
    "sparse_cdist",
    "vertex_cover",
    "earth_movers_distance",
    "matching_divergence",
    "neighborhood_preservation",
]
