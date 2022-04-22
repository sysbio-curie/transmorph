#!/usr/bin/env python3

from typing import List, Type

from ..watchers import Watcher


class IsWatchable:
    """
    A watchable object is an object that can be observed by a Watcher.
    """

    def __init__(self, compatible_watchers: List[Type]):
        self.compatible_watchers = compatible_watchers
        self.watchers: List[Watcher] = []

    def is_compatible(self, watcher: Watcher) -> bool:
        """
        Checks if a watcher is compatible with the object.
        """
        return (
            isinstance(watcher, tuple(self.compatible_watchers))
            and watcher not in self.watchers
        )

    def add_watcher(self, watcher: Watcher) -> None:
        """
        Adds a watcher to the layer to monitor it. Only the Watcher
        class should call this function, and is trusted to do so.
        """
        assert self.is_compatible(watcher)
        self.watchers.append(watcher)

    def update_watchers(self) -> None:
        """
        Calls the different watchers of the object.
        """
        for watcher in self.watchers:
            watcher.compute()
