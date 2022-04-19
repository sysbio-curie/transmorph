#!/usr/bin/env python3

from __future__ import annotations

import logging
import numpy as np
import warnings

from abc import ABC, abstractmethod

from anndata import AnnData
from transmorph import logger
from typing import Any, Callable, List, Optional, Type

from .layers import LayerMatching
from ..stats import edge_accuracy


class IsWatchable:
    """
    A watchable object is an object that can be observed by a Watcher.
    """

    def __init__(self, compatible_watchers: List[Type]):
        self.compatible_watchers = compatible_watchers
        self.watchers: List[Watcher] = []

    def add_watcher(self, watcher: Watcher) -> None:
        """
        Adds a watcher to the layer to monitor it. Only the Watcher
        class should call this function, and is trusted to do so.
        """
        assert watcher not in self.watchers
        assert isinstance(watcher, tuple(self.compatible_watchers))
        self.watchers.append(watcher)

    def update_watchers(self, datasets: List[AnnData]) -> None:
        """
        Calls the different watchers of the object.
        """
        for watcher in self.watchers:
            watcher.compute(datasets)


class Watcher(ABC):
    """
    Watchers watch functional layers functioning.
    They are not part of the functional pipeline, but can be
    used to monitor the behaviour of functional layers. For instance
    they are handy to gather quality metrics in various steps of
    the pipeline. All Watcher implementation must inherit this class.

    Parameters
    ----------
    target: Layer
        Layer to watch.

    target_type: Type
        Type of layer that can be watched by the Watcher.

    Attributes
    ----------
    data: Dict
        Contains Watcher output as a dictionary, which is useful to
        allow different types of measurements to be expressed as the
        same data type.

    readable: bool
        Indicates if the watcher is ready to be read, meaning it has
        computed and formatted its data.
    """

    WatcherID = 0

    def __init__(self, target: IsWatchable):
        self.watcher_id = Watcher.WatcherID
        Watcher.WatcherID += 1
        assert isinstance(target, IsWatchable)
        self.target = target
        self.target.add_watcher(self)
        self.verbose = False
        self.data = {}
        self.readable = False

    def __str__(self):
        return f"WAT[{self.watcher_id}]"

    def _log(self, msg: str, level: int = logging.DEBUG):
        logger.log(level, f"{self} > {msg}")

    @abstractmethod
    def compute(self, datasets: List[AnnData]) -> None:
        pass

    def get(self, key: str) -> Any:
        """
        As Watchers are not supposed to be used for functional purposes,
        and data type is uncontrolled, this is an unsafe getter that
        tries to retrieve Watcher monitoring data. Returns None in case
        of abnormal usage. The user is responsible to check output validity
        for this method.
        """
        if not self.readable:
            warnings.warn("Watcher is not ready.")
            return None
        if key not in self.data:
            warnings.warn(
                f"Unknown key: {key}. Available keys are {list(self.data.keys())}"
            )
            return None
        return self.data[key]


class WatcherTiming(Watcher):
    """
    Watches a Layer, and measures the time it takes to execute.

    Parameters
    ----------
    target: Layer
        Layer to watch.
    """

    def __init__(
        self,
        target: IsWatchable,
    ):
        super().__init__(target)

    def __str__(self) -> str:
        return f"{super().__str__()} - TIM"

    def compute(self, datasets: List[AnnData]) -> None:
        self._log("Retrieving time elapsed...")
        self.readable = False
        self.data["time"] = self.target.get_time_spent()
        self.readable = True


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
        super().__init__(target, LayerMatching)
        if callback is None:
            callback = edge_accuracy
        self.callback = callback
        self.label = label
        self.ignore_unmatched = ignore_unmatched

    def __str__(self) -> str:
        return f"{super().__str__()} - MAT"

    def compute(self, datasets: List[AnnData]) -> None:
        self._log("Computing matching metrics...")
        assert type(self.target) is LayerMatching
        self.readable = False
        ndatasets = len(datasets)
        for i in range(ndatasets):
            src = datasets[i]
            # Retrieves subsampling size
            n_anchors_i = self.target.get_anchors(src).sum()
            self.data[f"#samples{i}"] = n_anchors_i
            for j in range(i + 1, ndatasets):
                ref = datasets[j]
                Tij = self.target.get_matching(src, ref)
                if not self.ignore_unmatched:
                    n_anchors_j = self.target.get_anchors(ref).sum()
                    n_anchors = n_anchors_i + n_anchors_j
                else:
                    counts1 = np.array(Tij.sum(axis=1))[:, 0]
                    counts2 = np.array(Tij.sum(axis=0))[0]
                    n_anchors = (counts1 != 0).sum() + (counts2 != 0).sum()
                self.data[f"{i},{j}"] = self.callback(
                    src, ref, Tij, self.label, n_anchors
                )
        self.readable = True
