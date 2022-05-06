#!/usr/bin/env python3

from __future__ import annotations

import numpy as np

from anndata import AnnData
from scipy.sparse import csr_matrix
from typing import Callable, Dict, List, Optional, Tuple

from . import Layer
from ..matching import Matching, _TypeMatchingSet
from ..subsampling import Subsampling
from ..traits import (
    CanCatchChecking,
    ContainsTransformations,
    IsProfilable,
    profile_method,
    IsSubsamplable,
    IsRepresentable,
    UsesSampleLabels,
)
from ..traits.utils import preprocess_traits


class LayerMatching(
    Layer,
    CanCatchChecking,
    ContainsTransformations,
    IsProfilable,
    IsSubsamplable,
):
    """
    A LayerMatching encapsulates a matching algorithm, used to assess
    similarity between samples across datasets. It then stores matching
    results internally, and provides them upon request to a LayerMerging.
    Merings use this information to build a common embedding between
    datasets. Temporary transformations can be loaded in LayerMatching
    to be carried out before the matching algorithm.

    Parameters
    ----------
    matching: Matching
        Matching algorithm contained in the layer. This object is
        endowed with a fit() method, that will be called by the
        layer.

    subsampling: Optional[Subsampling], default = None
        Subsampling algorithm to use before the matching, can help
        for performance when dealing with large datasets. Note it
        tends to greatly reduce the number of matching edges.

    evaluators: List[Tuple[str, Callable]], default = []
        List of evaluation metrics f : AnnData, AnnData, csr_matrix -> float,
        endowed with a string key. They can then be accessed via get_metric(str, i, j).

    Attributes
    ----------
    matching_matrices: _TypeMatchingSet = Dict[Tuple[int, int], csr_matrix]
        Internal matching matrices, stored one internal matching has
        been called. Contains in coordinates (i, j) the matching
        between datasets i and j stored as a CSR matrix. These matrices
        can be provided to LayerMerging, and must not be directly
        modified.
    """

    def __init__(
        self,
        matching: Matching,
        subsampling: Optional[Subsampling] = None,
        evaluators: List[Tuple[str, Callable]] = [],
    ) -> None:
        Layer.__init__(
            self,
            compatible_inputs=[IsRepresentable],
            str_identifier="MATCHING",
        )
        CanCatchChecking.__init__(self)
        ContainsTransformations.__init__(self)
        IsProfilable.__init__(self)
        IsSubsamplable.__init__(self, subsampling)
        self.matching = matching
        self.matching_matrices: Optional[_TypeMatchingSet] = None
        assert all(
            isinstance(ev, Tuple) for ev in evaluators
        ), "Expected evaluators to be provided as (name: str, f: Callable)."
        self.evaluators = evaluators
        self.evaluator_results: Dict[str, np.ndarray] = {}

    @profile_method
    def fit(self, datasets: List[AnnData]) -> List[Layer]:
        """
        Runs the internal algorithm after carrying out the
        appropriate preprocessings. Then, returns next layers
        in the model.

        Parameters
        ----------
        datasets: List[AnnData]
            Datasets to run matching on.
        """
        self.datasets = datasets.copy()  # Keeping a copy to preserve order

        self.log(f"Retrieving data from {self.embedding_reference.repr_key}.")
        # Preprocessing
        Xs = self.transform(
            datasets=datasets,
            representer=self.embedding_reference,
            log_callback=self.log,
        )

        # Loading anndata information
        is_feature_space = (
            self.embedding_reference.is_feature_space and self.preserves_space
        )
        assert is_feature_space is not None
        preprocess_traits(self.matching, datasets, is_feature_space)

        # Subsampling
        if self.has_subsampling:
            self.info("Subsampling datasets...")
        self.compute_subsampling(
            datasets=datasets,
            matrices=Xs,
            is_feature_space=is_feature_space,
            log_callback=self.log,
        )
        Xs = self.subsample_matrices(matrices=Xs)
        if isinstance(self.matching, UsesSampleLabels):
            # FIXME this will cause trouble if we update USL trait
            self.matching.labels = self.subsample_matrices(self.matching.labels)

        self.matching.check_input(Xs)

        # Matching then supersampling matrices
        self.info(f"Calling matching {self.matching}.")
        self.matching_matrices = {}
        for key, T in self.matching.fit(Xs).items():
            i, j = key
            T = self.unsubsample_matrix(T, i, j)
            assert isinstance(T, csr_matrix)
            self.matching_matrices[i, j] = T
            self.log(f"Datasets {key}, found {T.data.shape[0]} edges.")

        # Running evaluators if needed
        ndatasets = len(datasets)
        if len(self.evaluators) > 0:
            self.log(f"Running {len(self.evaluators)} matching evaluators.")
        for key, evaluator in self.evaluators:
            results = np.zeros((ndatasets, ndatasets), dtype=np.float32)
            for i, adata_i in enumerate(datasets):
                for j, adata_j in enumerate(datasets):
                    if i >= j:  # Evaluators are assumed symmetrical
                        continue
                    results[i, j] = results[j, i] = evaluator(
                        adata_i,
                        adata_j,
                        self.matching_matrices[i, j],
                    )
            self.evaluator_results[key] = results

        # Trimming? Extrapolating?
        return self.output_layers

    def get_matchings(self) -> _TypeMatchingSet:
        """
        Returns computed matchings for read-only purposes.
        get_matchings()[i, j] is the matching between datasets
        i and j.
        """
        assert self.matching_matrices is not None, "Layer is not fit."
        return self.matching_matrices

    def get_matching_eval(self, evaluator: str) -> np.ndarray:
        """
        Returns the evaluation result of matching between two datasets indices.

        Parameters
        ----------
        evaluator: str
            Identifier of the evaluator to choose.
        """
        assert evaluator in self.evaluator_results, f"Unknown evaluator: {evaluator}"
        return self.evaluator_results[evaluator].copy()

    def get_pairwise_matching_eval(self, evaluator: str, i: int, j: int) -> float:
        """
        Returns the evaluation result of matching between two datasets indices.

        Parameters
        ----------
        evaluator: str
            Identifier of the evaluator to choose.

        i: int
            Index of the first dataset, as provided in fit()

        j: int
            Index of the second dataset, as provided in fit()
        """
        return self.get_matching_eval(evaluator)[i, j]
