#!/usr/bin/env python3

from __future__ import annotations

from abc import ABC, abstractmethod
from anndata import AnnData
from typing import List, Optional, Type

from ..traits import CanLog, IsRepresentable, assert_trait
from ...utils.type import assert_type


class Layer(ABC, CanLog):
    """
    A Layer wraps an integration module, and manage its connections
    with other modules. All Layers derive from this class, and can be
    enriched using traits.
    """

    # Provides a unique ID to each layer
    LayerID = 0

    def __init__(
        self,
        compatible_inputs: List[Type] = [],
        str_identifier: str = "BASE",
    ) -> None:
        CanLog.__init__(self, str_identifier=f"LAYER_{str_identifier}#{Layer.LayerID}")
        self.layer_id = Layer.LayerID
        Layer.LayerID += 1
        self.compatible_inputs = compatible_inputs
        self.input_layer: Optional[Layer] = None
        self.output_layers: List[Layer] = []
        self.profiler = None
        self._embedding_reference = None
        self.time_elapsed = -1
        self.log("Initialized.")

    def connect(self, layer: Layer) -> None:
        """
        Connects the current layer to an output layer, if compatible.

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
        if layer.embedding_reference is None:
            if not isinstance(self, IsRepresentable):
                reference = self.embedding_reference
            else:
                reference = self
            self.log(f"{reference} chosen as default embedding reference for {self}.")
            layer.embedding_reference = reference

    @abstractmethod
    def fit(self, datasets: List[AnnData]) -> List[Layer]:
        """
        This is the computational method, running an internal module.
        It returns a list of downstream layers, to call next.

        Parameters
        ----------
        datasets: List[AnnData]
            List of AnnData datasets to process.
        """
        pass

    @property
    def embedding_reference(self) -> IsRepresentable:
        """
        Retrieves closest Representable object upstream from current layer.
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
        Sets a Representable object to be the one providing matrix
        representations of datasets.
        """
        assert_trait(reference, IsRepresentable)
        self._embedding_reference = reference
