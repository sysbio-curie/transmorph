#!/usr/bin/env python3

import numpy as np

from anndata import AnnData
from scipy.sparse import csr_matrix
from typing import List, Optional


class UsesSampleLabels:
    """
    Allows a class to extract sample labels.

    Parameters
    ----------
    label_obs: str
        AnnData obs key to extract label from. If left None,
        this trait is ignored.
    """

    def __init__(self, label_obs: Optional[str] = None) -> None:
        self.label_obs = label_obs
        self.labels: List[np.ndarray] = []
        self.fitted = False

    def _check_index(self, idx: int) -> None:
        """
        Raises an exception if self.labels[$idx] is not accessable.
        """
        if self.label_obs is None:
            raise RuntimeError("Label obs was not specified.")
        if not self.fitted:
            raise RuntimeError("self.retrieve_all_labels not called.")
        if idx >= len(self.labels):
            raise IndexError(
                f"Index {idx} out of range for list " f"of size {len(self.labels)}."
            )

    def retrieve_all_labels(self, datasets: List[AnnData]) -> None:
        """
        Retrieves and stores labels from AnnData objects.
        """
        if self.label_obs is None:
            return
        assert all(self.label_obs in adata.obs for adata in datasets), (
            f"Key '{self.label_obs}' missing in some datasets. Available label "
            f"keys in this dataset are: {', '.join(sorted(datasets[0].obs))}"
        )
        for adata in datasets:
            self.labels.append(adata.obs[self.label_obs].to_numpy())
        self.fitted = True

    def get_dataset_labels(self, idx: int) -> np.ndarray:
        """
        Returns labels of dataset #idx.
        """
        self._check_index(idx)
        return self.labels[idx]

    def get_matching_matrix(self, idx_1: int, idx_2: int) -> csr_matrix:
        """
        Returns a n1 x n2 matrix where M[i, j] = yi == yj, to the
        CSR sparse format.
        """
        labels_1 = self.get_dataset_labels(idx_1)
        labels_2 = self.get_dataset_labels(idx_2)
        return csr_matrix(labels_1[:, None] == labels_2)
