#!/usr/bin/env python3

import logging
import warnings

from typing import Type

from ..._logging import logger


class CanLog:
    """
    This trait allows a class to send messages to the logging system,
    and attributes it a name. It gives access to a set of logging
    methods that can then be used at any moment by the objects.

    Parameters
    ----------
    str_identifier: str
        String identifier of the object to use duing logging.
    """

    def __init__(self, str_identifier: str):
        self.str_identifier = str_identifier

    def log(self, msg: str, level: int = logging.DEBUG) -> None:
        """
        Transmits a message to the logging module.

        Parameters
        ----------
        msg: str
            Message to print

        leve: int, default = logging.DEBUG
            Message priority. Set it higher to make it pass filters.
        """
        logger.log(level, f"{self.str_identifier} > {msg}")

    def info(self, msg: str) -> None:
        """
        Transmits a message to the logging module with INFO priority.

        Parameters
        ----------
        msg: str
            Message to print
        """
        self.log(msg, level=logging.INFO)

    def warn(self, msg: str) -> None:
        """
        Emits a warning message that will both reach the logger and the warning
        console stream.

        Parameters
        ----------
        msg: str
            Message to print
        """
        from ..._settings import settings

        if settings.logging_level_console < logging.ERROR:
            warnings.warn(msg)
        self.log(msg)

    def raise_error(self, error_type: Type, msg: str = "") -> None:
        """
        Raises an error of the specified type, and prints the message both in
        the console and in the logging stream.
        """
        self.log(f"{error_type.__name__} -- {msg}")
        raise error_type(msg)

    def __str__(self) -> str:
        return self.str_identifier
