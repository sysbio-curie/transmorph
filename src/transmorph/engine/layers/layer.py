#!/usr/bin/env python3

from __future__ import annotations

from abc import ABC, abstractmethod
from anndata import AnnData
from typing import List, Optional, Type

from ..traits import CanLog, IsRepresentable, assert_trait
from ...utils.misc import assert_type


class Layer(ABC, CanLog):
    """
    A Layer is a structural object which wraps an algorithmic module,
    and manages connections with other layers. All Layers derive from
    this class, and can be enriched using traits. Layers are the deepest
    objects in the architecture allowed to manipulate AnnData objects
    without using traits.

    Layers are not supposed to be manipulated for non-developement purposes.

    Attributes
    ----------
    _embedding_reference: Optional[IsRepresentable], default = None
        Layer able to provide a representation of datasets that
        this layer will use as a reference.

    compatible_inputs: List[Type], default = []
        List of layer types that can connect this class of layer.

    input_layer: Optional[Layer], default = None
        Incoming layer if any.

    layer_id: int
        Unique integer identifier, facilitates debugging.

    output_layers: List[Layer], default = []
        List of layers that receive information from this layer.
    """

    # Provides a unique ID to each layer
    LayerID = 0

    def __init__(
        self,
        compatible_inputs: List[Type] = [],
        str_identifier: str = "BASE",
    ) -> None:
        CanLog.__init__(self, str_identifier=f"LAYER_{str_identifier}#{Layer.LayerID}")
        self.compatible_inputs = compatible_inputs
        self._embedding_reference: Optional[IsRepresentable] = None
        self.input_layer: Optional[Layer] = None
        self.layer_id = Layer.LayerID
        Layer.LayerID += 1
        self.output_layers: List[Layer] = []
        self.log("Initialized.")

    def connect(self, layer: Layer) -> None:
        """
        Connects the current layer to an output layer, after
        checking if they are compatible.

        Parameters
        ----------
        layer: Layer
            Output layer of compatible type.
        """
        assert_type(layer, Layer)
        assert_type(self, tuple(layer.compatible_inputs))
        assert layer not in self.output_layers, f"{self} already connected to {layer}."
        assert layer.input_layer is None, f"{layer} has already a predecessor."
        layer.input_layer = self
        self.output_layers.append(layer)
        self.log(f"Connected to layer {layer}.")
        # Setting default embedding reference if needed
        if layer._embedding_reference is None:
            if not isinstance(self, IsRepresentable):
                reference = self.embedding_reference
            else:
                reference = self
            self.log(f"{reference} chosen as default embedding reference for {layer}.")
            layer.embedding_reference = reference

    @abstractmethod
    def fit(self, datasets: List[AnnData]) -> List[Layer]:
        """
        This is the computational method, running an internal algorithm.
        It then returns a list of downstream layers, to call next. It is
        the deepest method in the architecture allowed to manipulate
        AnnData objects without using traits.

        Parameters
        ----------
        datasets: List[AnnData]
            List of AnnData datasets to process.
        """
        pass

    @property
    def embedding_reference(self) -> IsRepresentable:
        """
        Retrieves closest IsRepresentable layer upstream from current layer.
        """
        if self._embedding_reference is None:
            if self.input_layer is None:
                self.raise_error(
                    ValueError,
                    "Input layer is None. Please make sure the "
                    "pipeline contains at least an input layer.",
                )
            self._embedding_reference = self.input_layer.embedding_reference
        return self._embedding_reference

    @embedding_reference.setter
    def embedding_reference(self, reference: IsRepresentable) -> None:
        """
        Sets a IsRepresentable layer to be the one providing matrix
        representations of datasets.
        """
        assert_trait(reference, IsRepresentable)
        self._embedding_reference = reference
