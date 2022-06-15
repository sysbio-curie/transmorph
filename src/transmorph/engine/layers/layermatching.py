#!/usr/bin/env python3

from __future__ import annotations

from anndata import AnnData
from scipy.sparse import csr_matrix
from typing import List, Optional


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
from ...utils.graph import (
    prune_edges_supervised,
    prune_edges_unsupervised,
    count_total_matching_edges,
)


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

    obs_class: Optional[str], default = None
        Provides the AnnData.obs key where sample type is stored. If
        specified, matching edges between samples of different class
        are pruned.

    edge_strictness: float, default = 0.5
        Fraction of edges to keep during the pruning process. Decreasing
        this value will decrease the numbers of matching edges, but can
        help getting rid of edges between samples of different classes.

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
        obs_class: Optional[str] = None,
        edge_strictness: float = 0.5,
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
        self.obs_class = obs_class
        self.edge_strictness = edge_strictness

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

        # Preprocessing if any
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

        # if some labels are available, prune edges
        if self.obs_class is not None:
            labels = [datasets[i].obs[self.obs_class] for i in range(len(datasets))]
            self.matching_matrices = prune_edges_supervised(
                self.matching_matrices,
                labels,
            )
        else:
            # Selects the largest number of min_batches preserving at least
            # edge_strictness % of edges
            prunings = [self.matching_matrices]
            for min_batch in range(1, len(datasets) - 2):
                prunings.append(
                    prune_edges_unsupervised(
                        self.matching_matrices,
                        len(datasets),
                        min_patterns=min_batch,
                    )
                )
            chosen = prunings[0]
            threshold = count_total_matching_edges(prunings[0]) * self.edge_strictness
            for i, new_matchings in enumerate(prunings):
                count_edges = count_total_matching_edges(new_matchings)
                self.log(f"min_batchs = {i}, {count_edges} edges.")
                if count_edges > threshold:
                    chosen = new_matchings
                else:
                    self.log(f"Chose min_batchs = {i - 1}.")
                    break
            self.matching_matrices = chosen

        return self.output_layers

    def get_matchings(self) -> _TypeMatchingSet:
        """
        Returns computed matchings for read-only purposes.
        get_matchings()[i, j] is the matching between datasets
        i and j.
        """
        assert self.matching_matrices is not None, "Layer is not fit."
        return self.matching_matrices
