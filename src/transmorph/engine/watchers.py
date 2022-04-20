#!/usr/bin/env python3

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Type

from .profiler import IsProfilable
from .traits import CanLog, assert_trait


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

    def update_watchers(self) -> None:
        """
        Calls the different watchers of the object.
        """
        for watcher in self.watchers:
            watcher.compute()


class Watcher(ABC, CanLog):
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

    def __init__(self, target: IsWatchable, str_identifier: str = "DEFAULT"):
        CanLog.__init__(self, f"WATCHER_{str_identifier}#{Watcher.WatcherID}")
        self.watcher_id = Watcher.WatcherID
        Watcher.WatcherID += 1
        assert isinstance(target, IsWatchable)
        self.target = target
        self.target.add_watcher(self)
        self.verbose = False
        self.data = {}
        self.readable = False

    @abstractmethod
    def compute(self) -> None:
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
            self.warn("Watcher is not ready.")
            return None
        if key not in self.data:
            self.warn(
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
        super().__init__(target, str_identifier="TIMING")

    def compute(self) -> None:
        self.log("Retrieving time elapsed...")
        self.readable = False
        assert_trait(self.target, IsProfilable)
        self.data["time"] = self.target.get_time_spent()
        self.readable = True
