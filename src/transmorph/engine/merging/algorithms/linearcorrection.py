#!/usr/bin/env python3

import anndata as ad
import numpy as np

from scipy.sparse import csr_matrix
from typing import List

from ..merging import Merging
from ...traits.usesreference import UsesReference
from ....utils.graph import (
    get_nearest_vertex_from_set,
    nearest_neighbors,
    smooth_correction_vectors,
)
from ....utils.matrix import sort_sparse_matrix


class LinearCorrection(Merging, UsesReference):
    """
    LinearCorrection is a way to merge vectorized datasets embedded
    in the same vector space onto a reference, aiming to solve issues
    of barycentric merging with partial matchings and overfitting.
    LinearCorrection requires all datasets to be already embedded in
    a common features space.

    Starting from two datasets X (source) and Y (reference) and a
    row-normalized matching T where Tij > 0 iff xi and yj are matched,
    we compute correction vectors c(Xm) between matched points Xm and the
    barycenter of their matches,

    c(Xm) = bary_Y(Xm, Tm) - Xm

    We end up with a set of correction vectors c(Xm), and need to
    define c(Xu) for unmatched samples.

    Then, all matched points are associated with the correction vector of
    their nearest corrected neigbhbor.

    Parameters
    ----------
    n_neighbors: int, default = 10
        Number of neighbors to use to compute the correction extrapolation
        graph.

    transformation_rate: float, default = 1.0
        Output merging is interpolated as a linear combination of
        original embedding and merged embedding, with 0.0 being the
        original embedding and 1.0 being the merged embedding.
    """

    def __init__(self, n_neighbors: int = 10, transformation_rate: float = 1.0):
        Merging.__init__(
            self,
            preserves_space=True,
            str_identifier="LINEAR_CORRECTION",
            matching_mode="normalized",
            transformation_rate=transformation_rate,
        )
        UsesReference.__init__(self)
        self.n_neighbors = n_neighbors

    def check_input(self, datasets: List[np.ndarray]) -> None:
        """
        Checks if all datasets are of same dimensionality.
        """
        dref = datasets[0].shape[1]
        assert np.all(
            [X.shape[1] == dref for X in datasets]
        ), "All datasets must be embedded in the same space to perform LinearCorrection"

    def project(
        self,
        X_src: np.ndarray,
        X_ref: np.ndarray,
        nn_src: csr_matrix,
        T: csr_matrix,
    ) -> np.ndarray:
        """
        Returns the projected view of X onto Y given the matching T
        """
        nsamples = X_src.shape[0]

        corrected_idx = np.array(T.sum(axis=1))[:, 0] > 0
        ncorrected = sum(corrected_idx)
        self.log(f"Corrected samples: {ncorrected} ({int(100*ncorrected/nsamples)}%).")

        ref_locations = T @ X_ref
        corr_vectors = np.zeros(X_src.shape, dtype=np.float32)
        corr_vectors[corrected_idx] = (
            ref_locations[corrected_idx] - X_src[corrected_idx]
        )

        indices, distances = sort_sparse_matrix(nn_src)
        references = get_nearest_vertex_from_set(indices, distances, corrected_idx)
        unreferenced = references == -1
        nunreferenced = sum(unreferenced)
        nreferenced = nsamples - nunreferenced - ncorrected
        self.log(
            f"Newly corrected samples: {nreferenced} "
            f"(+{int(100*nreferenced/nsamples)}%)."
        )
        self.log(
            f"Unreferenced samples: {nunreferenced} "
            f"({int(100*nunreferenced/nsamples)}%)."
        )
        if nunreferenced / nsamples > 0.1:
            self.warn(
                "More than 10% of samples are not matched, and are disconnected "
                "from any matched samples. Please make sure the reference dataset "
                "is comprehensive enough, and number of neighbors is high enough."
            )
        references[unreferenced] = np.arange(X_src.shape[0])[unreferenced]
        corr_vectors = smooth_correction_vectors(
            X_src,
            corr_vectors[references],
            indices,
            distances,
        )

        return X_src + corr_vectors * self.transformation_rate

    def transform(
        self, datasets: List[ad.AnnData], embeddings: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Computes correction vectors, then transforms.
        """
        k_ref = self.reference_index
        assert k_ref is not None, "No reference provided."
        X_ref = self.get_reference_item(embeddings)
        assert X_ref is not None, "No reference provided."
        result = []
        for k, (adata, X) in enumerate(zip(datasets, embeddings)):
            if X is X_ref:
                result.append(X_ref)
                continue
            T = self.get_matching(k, k_ref)
            nn_src = nearest_neighbors(
                adata,
                mode="distances",
                n_neighbors=self.n_neighbors,
            )
            projection = self.project(X, X_ref, nn_src, T)
            result.append(projection)
        return result
