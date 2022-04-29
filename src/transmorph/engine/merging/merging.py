#!/usr/bin/env python3

from ..traits.usesmatching import UsesMatching, _TypeMatchingModes
from ..transforming import Transformation


class Merging(Transformation, UsesMatching):
    """
    A Merging is an algorithm that is used to embed several datasets
    in a joint space, combining dataset geometry and inter-dataset
    matchings. In practice, it takes a set of matrix representations
    as input and returns a set of matrix representations in output,
    which justifies its Transformation inheritance. All Merging
    algorithms must inherit Merging.

    There are two main merging types: (1) mergings which operate in
    a space where datasets are already embedded, shifting samples in
    order to bring them together. (2) mergings which learn a new
    embedding space for datasets.

    For mergings of type (1), it is possible to use a transformation
    rate coefficient, which determines by how much samples of moved,
    0.0 meaning no movement at all and 1.0 meaning complete movement.
    This parameter can be used to design pipelines with iterative
    matchings and mergings.

    Parameters
    ----------
    preserves_space: bool, default = False
        Determines if the merging preserves both the embedding space
        and its basis.

    str_identifier: str, default = "DEFAULT"
        Merging name, for logging purposes

    matching_mode: Literal["raw", "normalized", "bool"]
        Determines the preprocessing to apply to input matching
        matrices.

        - raw: Matching matrices are left untouched
        - normalized: Matching matrices are row-normalized to sum
          to 1.
        - bool: Matching matrices are simplified to 1 = match and
          0 = no match.

    transformation_rate: float, default = 1.0
        This parameter only applies if the merging preserves space.
        Output merging is then interpolated as a linear combination of
        original embedding and merged embedding, with 0.0 being the
        original embedding and 1.0 being the merged embedding.
    """

    def __init__(
        self,
        preserves_space: bool = False,
        str_identifier: str = "DEFAULT",
        matching_mode: _TypeMatchingModes = "raw",
        transformation_rate: float = 1.0,
    ):
        Transformation.__init__(
            self,
            preserves_space=preserves_space,
            str_identifier=f"MERGING_{str_identifier}",
            transformation_rate=transformation_rate,
        )
        # Removes TRANSFORMATION_
        self.str_identifier = f"MERGING_{str_identifier}"
        UsesMatching.__init__(self, mode=matching_mode)
