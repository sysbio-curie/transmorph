#!/usr/bin/env python3

from .watcher import Watcher
from .watchermatching import WatcherMatching
from .watchertiming import WatcherTiming
from ..traits import IsWatchable


def can_watch(watcher: Watcher, target: IsWatchable) -> bool:
    """
    Checks if a watcher is compatible wit a target.
    """
    if not isinstance(target, IsWatchable):
        return False
    return target.is_compatible(watcher)


__all__ = ["Watcher", "WatcherMatching", "WatcherTiming"]
