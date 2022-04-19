#!/usr/bin/env python3

import logging
from typing import Literal

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

    @property
    def verbose(self) -> str:
        if self.logging_level_console <= logging.DEBUG:
            return "DEBUG"
        if self.logging_level_console <= logging.INFO:
            return "INFO"
        if self.logging_level_console <= logging.WARNING:
            return "WARNING"
        return "ERROR"

    @verbose.setter
    def verbose(self, level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]) -> None:
        if level == "DEBUG":
            int_level = logging.DEBUG
        elif level == "INFO":
            int_level = logging.INFO
        elif level == "WARNING":
            int_level = logging.WARNING
        elif level == "ERROR":
            int_level = logging.ERROR
        else:
            raise ValueError(level)
        self.logging_level_console = int_level


settings = TransmorphSettings()
