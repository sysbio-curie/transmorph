#!/usr/bin/env python3

from __future__ import annotations

import anndata as ad
import numpy as np

from abc import ABC, abstractmethod
from typing import List

from ..traits import CanLog

_TypeSubsampling = np.ndarray


class Subsampling(ABC, CanLog):
    """
    A subsampling scheme choses a subset of representers from a large collection
    of points. These representers are then used during heavy computational parts
    of the pipeline to speed up computations. SubsamplingABC is the abstraction
    that allows other parts of Transmorph to manipulate subsampling schemes.

    Parameters
    ----------
    str_type: str
        String representation of the matching algorithm. Will
        typically be the matching algorithm name.
    """

    def __init__(self, str_identifier: str = "DEFAULT"):
        CanLog.__init__(self, str_identifier=f"SUBSAMPLING_{str_identifier}")

    def check_input(self, datasets: List[np.ndarray]) -> None:
        """
        Takes a list of np.ndarray representing datasets to subsample,
        and verifies their validity. Should raise warnings or
        errors in case of unexpected input. Will be called before
        carrying out the matching task. Can be overrode by child
        classes.
        """
        pass

    @abstractmethod
    def subsample(
        self, datasets: List[ad.AnnData], embeddings: List[np.ndarray]
    ) -> List[_TypeSubsampling]:
        """
        Applies a subsampling algorithm to a list of matrices
        representing datasets. Returns results in the following
        shape,
        List[
            (anchors_1, references_1),
            ...
            (anchors_N, references_N),
        ]

        Parameters
        ----------
        datasets: List[np.ndarray]
            Matrix representation of datasets to subsample.
        """
        pass
