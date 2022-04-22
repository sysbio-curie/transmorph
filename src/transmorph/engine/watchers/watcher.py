#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import Any

from ..traits import CanLog, IsWatchable


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

    TODO update all this
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
