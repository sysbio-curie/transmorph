#!/usr/bin/env python3

import logging

from ._logging import logger, _DEFAULT_LEVEL_FILE, _DEFAULT_LEVEL_CONSOLE
from .utils.type import assert_type


class TransmorphSettings:
    """
    Settings manager.
    """

    def __init__(self):
        self._logging_level_file = _DEFAULT_LEVEL_FILE
        self._logging_level_console = _DEFAULT_LEVEL_CONSOLE
        logger.debug("Transmorph settings initialized.")

    @property
    def logging_level(self) -> int:
        return min(self.logging_level_console, self.logging_level_file)

    @logging_level.setter
    def logging_level(self, value: int) -> None:
        self.logging_level_console = value
        self.logging_level_file = value

    @property
    def logging_level_file(self) -> int:
        return self._logging_level_file

    @logging_level_file.setter
    def logging_level_file(self, value: int) -> None:
        assert_type(value, int)
        self._logging_level = value
        logger.setLevel(value)
        for handler in logger.handlers:
            if type(handler) is logging.FileHandler:
                handler.setLevel(value)
        logger.debug(f"Setting file logger level to {value}.")

    @property
    def logging_level_console(self) -> int:
        return self._logging_level_console

    @logging_level_console.setter
    def logging_level_console(self, value: int) -> None:
        assert_type(value, int)
        self._logging_level = value
        logger.setLevel(value)
        for handler in logger.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(value)
        logger.debug(f"Setting console logger level to {value}.")


settings = TransmorphSettings()
