#!/usr/bin/env python3

import numpy as np

from typing import List, Literal

from ..matching import Matching, _TypeMatchingSet


class CombineMatching(Matching):
    def __init__(self, rule: Literal["and", "or", "min", "max"], *matchings):
        Matching.__init__(self, str_identifier="COMBINE_MATCHING")
        assert rule in [
            "and",
            "or",
            "min",
            "max",
        ], f"Unrecognized rule: {rule}. Expected 'and', 'or', 'min' or 'max'."
        self.rule = rule
        self.matchings: List[Matching] = [matching for matching in matchings]
        assert all(
            isinstance(matching, Matching) for matching in self.matchings
        ), "Non-matching object detected."

    def check_input(self, datasets: List[np.ndarray]) -> None:
        for matching in self.matchings:
            matching.check_input(datasets)

    def fit(self, datasets: List[np.ndarray]) -> _TypeMatchingSet:
        matching_results: List[_TypeMatchingSet] = []
        for matching in self.matchings:
            matching_results.append(matching.fit(datasets))
        result = {}
        for (i, j) in matching_results[0]:
            result_mtx = matching_results[0][i, j].copy()
            for matching_result in matching_results[1:]:
                if self.rule == "and":
                    result_mtx = result_mtx * matching_result[i, j]
                elif self.rule == "or":
                    result_mtx = result_mtx + matching_result[i, j]
                elif self.rule == "min":
                    result_mtx = result_mtx.minimum(matching_result)[i, j]
                elif self.rule == "max":
                    result_mtx = result_mtx.maximum(matching_result)[i, j]
                else:
                    raise ValueError(f"Unrecognized rule: {self.rule}")
            result[i, j] = result_mtx
        return result
