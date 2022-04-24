#!/usr/bin/env python3

from anndata import AnnData
from typing import List

from transmorph.engine.traits import UsesReference

from . import Layer, LayerMatching
from ..merging import Merging
from ..traits import (
    ContainsTransformations,
    IsProfilable,
    profile_method,
    IsRepresentable,
    HasMetadata,
)


class LayerMerging(
    Layer,
    ContainsTransformations,
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
        IsProfilable.__init__(self)
        IsRepresentable.__init__(self, repr_key=f"{self}_{self.layer_id}")
        self.merging = merging

    @profile_method
    def fit(self, datasets: List[AnnData]) -> List[Layer]:
        """
        Runs preprocessings, then delegate to the internal merging.
        """
        # Pleases the type checker
        Xs = self.transform(
            datasets=datasets,
            representer=self.embedding_reference,
            log_callback=self.info,
        )
        self.info(f"Running merging {self.merging}...")
        if isinstance(self.merging, HasMetadata):
            self.merging.retrieve_all_metadata(datasets)
        if isinstance(self.merging, UsesReference):
            self.merging.retrieve_reference_index(datasets)
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
        return self.output_layers
