#!/usr/bin/env python3

from __future__ import annotations

from typing import Dict

from .utils import assert_trait
from ..._profiling import profiler


def profile_method(method):
    # Profiling decorator for class methods that
    # allows to measure time elapsed in a profilable
    # object.
    def wrapper(*args):
        self = args[0]
        assert self is not None
        assert_trait(self, IsProfilable)
        tstr = f"{str(self)}.{method.__name__}"
        tid = profiler.task_start(tstr)
        result = method(*args)
        elapsed = profiler.task_end(tid)
        self.elapsed[tstr] = elapsed
        return result

    return wrapper


class IsProfilable:
    """
    A profilable object can be monitored by the Profiler. Methods
    to profile must be decorated by @profile_method.
    TODO: add a self.set_stop() allowing to monitor particular
    function parts
    """

    def __init__(self):
        self.elapsed: Dict[str, float] = {}

    def get_time_spent(self) -> float:
        """
        Returns the total time of all methods profiled.
        """
        return sum(self.elapsed.values())
