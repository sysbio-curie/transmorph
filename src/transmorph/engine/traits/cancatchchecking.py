#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from .isrepresentable import IsRepresentable

if TYPE_CHECKING:
    from ..layers import LayerChecking


class CanCatchChecking:
    """
    This trait allows a Layer to be the target of
    the rejected connection of a CheckingLayer.
    WARNING: Only layers should inherit this trait,
    as it uses layer attributes.
    """

    def __init__(self):
        # Handles two internal references. The first is
        # used in most cases, the second is used when called
        # by a checking layer that rejected its test.
        self.base_reference: Optional[IsRepresentable] = None
        self.check_reference: Optional[IsRepresentable] = None
        self._called_by_checking = False

    @property
    def called_by_checking(self) -> bool:
        return self._called_by_checking

    @called_by_checking.setter
    def called_by_checking(self, state: bool) -> None:
        """
        Toggles embedding layer to use.
        """
        self._called_by_checking = state
        if state:
            self._embedding_reference = self.check_reference
        else:
            self._embedding_reference = self.base_reference

    def catch_checking_rejected(self, layer: LayerChecking) -> None:
        """
        Registers a CheckingLayer.
        """
        assert isinstance(layer, IsRepresentable)
        self.base_reference = self._embedding_reference
        self.check_reference = layer
