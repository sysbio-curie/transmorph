#!/usr/bin/env python3

import logging
from typing import Any, Dict, Literal, Optional

from ._logging import logger, _DEFAULT_LEVEL_FILE, _DEFAULT_LEVEL_CONSOLE
from .utils.type import assert_type


class TransmorphSettings:
    """
    Settings manager.
    """

    def __init__(self):
        # Logging
        self._logging_level_file: int = _DEFAULT_LEVEL_FILE
        self._logging_level_console: int = _DEFAULT_LEVEL_CONSOLE
        # Neighbors
        self._n_neighbors: int = 15
        self._neighbors_algorithm: Literal["auto", "sklearn", "nndescent"] = "auto"
        self.neighbors_include_self_loops: bool = False
        self.neighbors_metric: str = "sqeuclidean"
        self._neighbors_metric_kwargs: Dict[str, Any] = {}
        self._neighbors_n_pcs: Optional[int] = 30
        self.neighbors_random_seed: int = 42
        self.neighbors_symmetrize: bool = False
        self.neighbors_use_scanpy: bool = True
        # Scale
        self.large_dataset_threshold: int = 2048
        # End
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

    @property
    def n_neighbors(self) -> int:
        return self._n_neighbors

    @n_neighbors.setter
    def n_neighbors(self, n: int) -> None:
        n = int(n)
        assert n > 0, f"Invalid number of neighbors {n}"
        self._n_neighbors = n

    @property
    def neighbors_algorithm(self) -> Literal["auto", "sklearn", "nndescent"]:
        return self._neighbors_algorithm

    @neighbors_algorithm.setter
    def neighbors_algorithm(
        self, algorithm: Literal["auto", "sklearn", "nndescent"]
    ) -> None:
        assert algorithm in ("auto", "sklearn", "nndescent")
        self._neighbors_algorithm = algorithm

    @property
    def neighbors_metric_kwargs(self) -> Dict:
        return self._neighbors_metric_kwargs

    @neighbors_metric_kwargs.setter
    def neighbors_metric_kwargs(self, kwargs: Optional[Dict]) -> None:
        if kwargs is None:
            kwargs = {}
        assert isinstance(kwargs, Dict)
        self._neighbors_metric_kwargs = kwargs

    @property
    def neighbors_n_pcs(self) -> Optional[int]:
        return self._neighbors_n_pcs

    @neighbors_n_pcs.setter
    def neighbors_n_pcs(self, n: Optional[int]) -> None:
        if n is None:
            self._neighbors_n_pcs = n
            return
        n = int(n)
        assert n > 0, f"Invalid number of pcs {n}."
        self._neighbors_n_pcs = n


settings = TransmorphSettings()
