#!/usr/bin/env python3

from anndata import AnnData
from typing import List

from . import Layer, LayerMatching
from ..merging import Merging
from ..traits import (
    ContainsTransformations,
    IsProfilable,
    IsRepresentable,
    IsWatchable,
    HasMetadata,
)
from ..watchers import WatcherTiming
from ... import profile_method


class LayerMerging(
    Layer,
    ContainsTransformations,
    IsWatchable,
    IsProfilable,
    IsRepresentable,
):
    """
    This layer performs a merging between two or more datasets and their matchings.
    It wraps an object derived from MergingABC.
    """

    def __init__(self, merging: Merging) -> None:
        Layer.__init__(
            self, compatible_inputs=[LayerMatching], str_identifier="MERGING"
        )
        ContainsTransformations.__init__(self)
        IsWatchable.__init__(self, compatible_watchers=[WatcherTiming])
        IsProfilable.__init__(self)
        IsRepresentable.__init__(self, repr_key=f"{self}_{self.layer_id}")
        self.merging = merging

    @profile_method
    def fit(self, datasets: List[AnnData]) -> List[Layer]:
        """
        Runs preprocessings, then delegate to the internal merging.
        """
        # Pleases the type checker
        if self.has_transformations:
            self.info("Calling preprocessings.")
        Xs = self.transform(datasets, self.embedding_reference)
        self.info("Running merging...")
        if isinstance(self.merging, HasMetadata):
            self.merging.retrieve_all_metadata(datasets)
        assert isinstance(self.input_layer, LayerMatching)
        self.merging.set_matchings(self.input_layer.get_matchings())
        Xs_transform = self.merging.transform(Xs)
        is_feature_space = (
            self.embedding_reference.is_feature_space  # Original matrices
            and self.preserves_space  # Internal transformations
            and self.merging.preserves_space  # Internal matching
        )
        for adata, X_after in zip(datasets, Xs_transform):
            self.write_representation(adata, X_after, is_feature_space=is_feature_space)
        self.info("Fitted.")
        return self.output_layers
