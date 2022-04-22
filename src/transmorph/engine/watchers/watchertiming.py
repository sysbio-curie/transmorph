#!/usr/bin/env python3

from . import Watcher
from ..traits import IsProfilable, IsWatchable, assert_trait


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
