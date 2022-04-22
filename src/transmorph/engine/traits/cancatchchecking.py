#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from .isrepresentable import IsRepresentable

if TYPE_CHECKING:
    from ..layers import Layer, LayerChecking


class CanCatchChecking:
    """
    This trait allows an object to be the target of
    the rejected connection of a CheckingLayer.
    """

    def __init__(self):
        self.layer_checking: Optional[LayerChecking] = None
        self._called_by_checking = False
        self.previous_embedding_reference: Optional[IsRepresentable] = None

    @property
    def called_by_checking(self) -> bool:
        return self._called_by_checking

    @called_by_checking.setter
    def called_by_checking(self) -> None:
        self._called_by_checking = True
        assert isinstance(self, Layer)
        assert isinstance(self.layer_checking, IsRepresentable)
        self.previous_embedding_reference = self.embedding_reference
        self.embedding_reference = self.layer_checking

    def restore_previous_mapping(self) -> None:
        self.embedding_reference = self.previous_embedding_reference

    def connect_rejected(self, layer: LayerChecking) -> None:
        """
        Registers a CheckingLayer.
        """
        self.layer_checking = layer
