#!/usr/bin/env python3

import logging

from numpy.random.mtrand import RandomState
from sklearn.utils import check_random_state
from typing import Any, Dict, Literal, Optional, TypeVar

from ._logging import logger
from .utils.misc import assert_type

T = TypeVar("T")


def use_setting(value: Optional[T], setting: T) -> T:
    """
    If value is None, returns setting, otherwise returns value.
    """
    if value is None:
        return setting
    return value


class TransmorphSettings:
    """
    Settings manager.
    TODO check all settings that are at risk
    TODO update random generation numpy
    """

    def __init__(self):

        # Logging
        self._logging_level_console: int = logging.INFO
        self._logging_level_file: int = logging.DEBUG
        self.logging_level_console = self._logging_level_console
        self.logging_level_file = self._logging_level_file

        # General
        self.global_random_seed: int = 42

        # Neighbors
        self._n_neighbors_max_init: int = 50
        self._n_neighbors_max: int = self._n_neighbors_max_init
        self._neighbors_algorithm: Literal["auto", "sklearn", "nndescent"] = "auto"
        self.neighbors_metric: str = "sqeuclidean"
        self._neighbors_metric_kwargs: Dict[str, Any] = {}
        self._neighbors_n_pcs: int = 15
        self.neighbors_random_seed: int = 42

        # UMAP
        self.umap_metric: str = "euclidean"
        self.umap_metric_kwargs: Dict[str, Any] = {}
        self.umap_min_dist: float = 0.5
        self.umap_spread: float = 1.0
        self.umap_maxiter: Optional[int] = None  # Automatic chhoice
        self.umap_alpha: float = 1.0
        self.umap_gamma: float = 1.0
        self.umap_negative_sample_rate: int = 5
        self.umap_a: Optional[float] = None
        self.umap_b: Optional[float] = None
        self.umap_random_state: RandomState = check_random_state(
            self.global_random_seed
        )
        self.umap_n_neighbors: int = 15

        # MDE
        self.mde_initialization: str = "quadratic"
        self.mde_repulsive_fraction: float = 1.0
        self.mde_device: Literal["cpu", "cuda"] = "cpu"

        # Scale
        self.small_dataset_threshold: int = 100
        self.large_dataset_threshold: int = 4096
        self.low_dimensional_threshold: int = 5
        self.high_dimensional_threshold: int = 60
        self.large_number_edges: int = 10_000_000
        self.low_features_ratio_threshold: float = 0.005
        self.is_discrete_unique_thr: int = 50
        self.vertexcover_n_neighbors: int = 5

        # End
        logger.debug("Transmorph settings initialized.")

    @property
    def logging_level_file(self) -> int:
        return self._logging_level_file

    @logging_level_file.setter
    def logging_level_file(self, value: int) -> None:
        logger.debug(f"Changing console logging level to {value}.")
        assert_type(value, int)
        if value < min(self._logging_level_file, self._logging_level_console):
            logger.setLevel(value)
        self._logging_level_file = value
        for handler in logger.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(self._logging_level_console)
            if type(handler) is logging.FileHandler:
                handler.setLevel(self._logging_level_file)

    @property
    def logging_level_console(self) -> int:
        return self._logging_level_console

    @logging_level_console.setter
    def logging_level_console(self, value: int) -> None:
        logger.debug(f"Changing console logging level to {value}.")
        assert_type(value, int)
        if value < min(self._logging_level_file, self._logging_level_console):
            logger.setLevel(value)
        self._logging_level_console = value
        for handler in logger.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(self._logging_level_console)
            if type(handler) is logging.FileHandler:
                handler.setLevel(self._logging_level_file)

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
            raise ValueError(
                level,
                f"Unknown verbose level {level}. Available options are "
                "'DEBUG', 'INFO', 'WARNING' or 'ERROR'.",
            )
        self.logging_level_console = int_level

    @property
    def n_neighbors_max(self) -> int:
        return self._n_neighbors_max

    @n_neighbors_max.setter
    def n_neighbors_max(self, n: int) -> None:
        n = int(n)
        assert n > 0, f"Invalid number of neighbors {n}"
        self._n_neighbors_max = n

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
    def neighbors_n_pcs(self) -> int:
        return self._neighbors_n_pcs

    @neighbors_n_pcs.setter
    def neighbors_n_pcs(self, n: int) -> None:
        n = int(n)
        assert n > 0, f"Invalid number of pcs {n}."
        self._neighbors_n_pcs = n


settings = TransmorphSettings()
