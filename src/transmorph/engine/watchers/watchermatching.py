#!/usr/bin/env python3

from anndata import AnnData
from typing import Optional

from ..watchers import Watcher


class WatcherMatching(Watcher):
    """
    Watches a LayerMatching, and measures its quality by comparing
    guessed edges with ground truth (matching between sample labels).

    Parameters
    ----------
    target: LayerMatching
        Layer containing the Matching to watch.

    label: str
        String identifying the AnnData.obs column containing label
        to compare.

    callblack: Callable, Optional
        Function to call to evaluate quality, must have the following
        signature:

        callback(AnnData, AnnData, csr_matrix, str, bool) -> float
        callback(adata_src, adata_ref, matching,
                 label, ignore_unmatched) -> float

        By default, the Watcher will use edge accuracy
        (see src/stats/matching.py)
    """

    def __init__(self):
        raise NotImplementedError

    def compute(self) -> None:
        """
        Retrieves information about the matching step:
        - Number of selected anchors for each dataset, accessible via
          self.get_n_samples(adata: AnnData)
        - Matching accuracy (or any other specified metric), accessible via
          self.get_accuracy(adata1: AnnData, adata2: AnnData)
        """
        raise NotImplementedError

    def get_n_samples(self, adata: AnnData) -> Optional[int]:
        """
        Returns number of subsampled samples in adata. Returns None if adata
        is not found.
        """
        raise NotImplementedError

    def get_accuracy(self, adata1: AnnData, adata2: AnnData) -> Optional[float]:
        """
        Returns matching accuracy between adata1 and adata2, or None if the
        matching is not found.
        """
        raise NotImplementedError
