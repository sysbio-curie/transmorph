#!/usr/bin/env python3

import numpy as np

from anndata import AnnData
from typing import Dict, List, Optional, Union
from umap.umap_ import simplicial_set_embedding, find_ab_params
from scipy.sparse import csr_matrix
from sklearn.utils import check_random_state
from warnings import warn

from ..matching.matchingABC import MatchingABC
from .mergingABC import MergingABC
from ..utils import extract_chunks
from ..utils.graph import combine_matchings, nearest_neighbors
from ..utils.anndata_interface import get_matrix


class MergingUMAP(MergingABC):
    def __init__(
        self,
        metric: str = "euclidean",
        metric_kwargs: Optional[Dict] = None,
        min_dist: float = 0.5,
        spread: float = 1.0,
        n_components: int = 2,
        maxiter: Optional[int] = None,
        alpha: float = 1.0,
        gamma: float = 1.0,
        negative_sample_rate: int = 5,
        a: Optional[float] = None,
        b: Optional[float] = None,
        concentration: float = 1.0,
        n_neighbors: int = 15,
        random_seed: int = 42,
    ):
        super().__init__(use_reference=False)
        self.n_components = n_components
        self.maxiter = maxiter
        self.alpha = alpha
        self.gamma = gamma
        self.neg_sr = negative_sample_rate
        if a is None or b is None:
            a, b = find_ab_params(spread, min_dist)
        self.a = a
        self.b = b
        self.metric = metric
        self.metric_kwargs = {} if metric_kwargs is None else metric_kwargs
        self.concentration = concentration
        self.n_neighbors = n_neighbors
        self.random_state = check_random_state(random_seed)

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
                    metric=self.metric,
                    metric_kwargs=self.metric_kwargs,
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
        nsamples = sum(adata.n_obs for adata in datasets)
        n_epochs = (
            self.maxiter
            if self.maxiter is not None
            else 500
            if nsamples < 10000
            else 200
        )
        X_result, _ = simplicial_set_embedding(
            data=None,
            graph=edges,
            n_components=self.n_components,
            initial_alpha=self.alpha,
            a=self.a,
            b=self.b,
            gamma=self.gamma,
            negative_sample_rate=self.neg_sr,
            n_epochs=n_epochs,
            init="spectral",
            random_state=self.random_state,
            metric=self.metric,
            metric_kwds=self.metric_kwargs,
            densmap=False,
            densmap_kwds={},
            output_dens=False,
            verbose=False,
        )
        return extract_chunks(X_result, [adata.n_obs for adata in datasets])
