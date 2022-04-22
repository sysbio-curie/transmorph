#!/usr/bin/env python3

from typing import Dict


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
