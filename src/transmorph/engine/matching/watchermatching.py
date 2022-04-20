#!/usr/bin/env python3

import numpy as np

from anndata import AnnData
from typing import Callable, Optional

from .layermatching import LayerMatching
from ..subsampling import IsSubsamplable
from ..watchers import Watcher
from ...stats import edge_accuracy


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

    def __init__(
        self,
        target: LayerMatching,
        label: str,
        ignore_unmatched: bool = False,
        callback: Optional[Callable] = None,
    ):
        super().__init__(target, str_identifier="MATCHING")
        if callback is None:
            callback = edge_accuracy
        self.callback = callback
        self.label = label
        self.ignore_unmatched = ignore_unmatched

    def compute(self) -> None:
        """
        Retrieves information about the matching step:
        - Number of selected anchors for each dataset, accessible via
          self.get_n_samples(adata: AnnData)
        - Matching accuracy (or any other specified metric), accessible via
          self.get_accuracy(adata1: AnnData, adata2: AnnData)
        """
        self.log("Computing matching metrics...")
        assert type(self.target) is LayerMatching
        self.readable = False
        for i, src in enumerate(self.target.datasets):
            n_anchors_i = IsSubsamplable.get_anchors(src).sum()
            self.data[f"#samples{i}"] = n_anchors_i
            for j, ref in enumerate(self.target.datasets):
                if j <= i:
                    continue
                Tij = self.target.get_matching(src, ref)
                if not self.ignore_unmatched:
                    n_anchors_j = IsSubsamplable.get_anchors(ref).sum()
                    n_anchors = n_anchors_i + n_anchors_j
                else:
                    counts1 = np.array(Tij.sum(axis=1))[:, 0]
                    counts2 = np.array(Tij.sum(axis=0))[0]
                    n_anchors = (counts1 != 0).sum() + (counts2 != 0).sum()
                self.data[f"{i},{j}"] = self.callback(
                    src, ref, Tij, self.label, n_anchors
                )
        self.readable = True

    def get_n_samples(self, adata: AnnData) -> Optional[int]:
        """
        Returns number of subsampled samples in adata. Returns None if adata
        is not found.
        """
        assert type(self.target) is LayerMatching
        try:
            idx = self.target.get_adata_index(adata)
        except ValueError:
            return None
        return self.data.get(f"#samples{idx}", None)

    def get_accuracy(self, adata1: AnnData, adata2: AnnData) -> Optional[float]:
        """
        Returns matching accuracy between adata1 and adata2, or None if the
        matching is not found.
        """
        assert type(self.target) is LayerMatching
        try:
            idx1 = self.target.get_adata_index(adata1)
            idx2 = self.target.get_adata_index(adata2)
        except ValueError:
            return None
        return self.data.get(f"{idx1},{idx2}", self.data.get(f"{idx2},{idx1}", None))
