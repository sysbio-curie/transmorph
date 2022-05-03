#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from .isrepresentable import IsRepresentable

if TYPE_CHECKING:
    from ..layers import LayerChecking


class CanCatchChecking:
    """
    A layer equipped with CanCatchChcking trait can be set as
    the rejected layer of a LayerChecking. It provides it a set
    of attributes and methods that allow their embedding reference
    to be safely manipulated to be temporarily set to the
    checking layer responsible for the rejection.

    Attributes
    ----------
    base_reference: IsRepresentable
        Initial embedding reference of the layer.

    check_reference: IsRepresentable
        Embedding reference of the checking layer, to use in case
        of rejection as an embedding reference.

    called_by_checking: boolean property
        Setting this attribute will switch representation
        accordingly between true embedding reference and checking
        reference.
    """

    def __init__(self):
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
        Registers a LayerChecking to be the one which can provide
        an alternate embedding representation.
        """
        assert isinstance(layer, IsRepresentable)
        self.base_reference = self._embedding_reference
        self.check_reference = layer
