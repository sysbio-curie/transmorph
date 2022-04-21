#!/usr/bin/env python3

from ..matching import UsesMatching, _TypeMatchingModes
from ..transforming import Transformation


class Merging(Transformation, UsesMatching):
    """
    A Merging is a particular class of Transformations that combine
    datasets with the help of matching information.
    """

    def __init__(
        self,
        preserves_space: bool = False,
        str_identifier: str = "DEFAULT",
        matching_mode: _TypeMatchingModes = "raw",
    ):
        Transformation.__init__(
            self,
            preserves_space=preserves_space,
            str_identifier=f"MERGING_{str_identifier}",
        )
        UsesMatching.__init__(self, mode=matching_mode)
