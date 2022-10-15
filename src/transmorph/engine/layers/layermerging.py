#!/usr/bin/env python3

from anndata import AnnData
from typing import List

from . import Layer, LayerMatching
from ..merging import Merging
from ..traits import (
    ContainsTransformations,
    IsProfilable,
    profile_method,
    IsRepresentable,
)
from ..traits.utils import preprocess_traits


class LayerMerging(
    Layer,
    ContainsTransformations,
    IsProfilable,
    IsRepresentable,
):
    """
    A LayerMerging encapsulates a merging algorithms, used to leverage
    matching information to embed several datasets in a common integration
    space. It can only follow a LayerMatching, and is able to provide
    matrix representation of AnnData datasets. Temporary transformations
    can be loaded in LayerMatching to be carried out before the matching
    algorithm.

    Parameters
    ----------
    merging: Merging
        Merging algorithm contained in the layer. This object is
        endowed with a transform() method, that will be called by the
        layer.
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
        Runs the internal algorithm after carrying out the
        appropriate preprocessings. Then, returns next layers
        in the model.

        Parameters
        ----------
        datasets: List[AnnData]
            Datasets to run merging on.
        """
        self.log(f"Retrieving data from {self.embedding_reference.repr_key}.")
        Xs = self.transform(
            datasets=datasets,
            representer=self.embedding_reference,
            log_callback=self.log,
        )
        is_feature_space = (
            self.embedding_reference.is_feature_space  # Original matrices
            and self.preserves_space  # Internal transformations
            and self.merging.preserves_space  # Internal matching
        )
        preprocess_traits(self.merging, datasets, is_feature_space)
        assert isinstance(self.input_layer, LayerMatching)
        self.merging.set_matchings(self.input_layer.get_matchings())
        self.info(f"Running merging {self.merging}...")
        Xs_transform = self.merging.transform(datasets, Xs)
        for adata, X_after in zip(datasets, Xs_transform):
            self.write_representation(adata, X_after, is_feature_space=is_feature_space)
        return self.output_layers
