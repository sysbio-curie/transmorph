#!/usr/bin/env python3

from typing import Dict, List, Union

import anndata as ad

from ._logging import logger
from ._profiling import profiler
from ._settings import settings, use_setting

__all__ = ["logger", "profiler", "settings", "use_setting"]

InputType = Union[List[ad.AnnData], Dict[str, ad.AnnData]]


def check_input_transmorph(datasets: InputType) -> None:
    # Simple input type checker for now
    # May be enhanced in the future
    if isinstance(datasets, List):
        assert all(isinstance(item, ad.AnnData) for item in datasets)
    elif isinstance(datasets, Dict):
        assert all(
            isinstance(key, str) and isinstance(value, ad.AnnData)
            for (key, value) in datasets.items()
        )
    else:
        raise TypeError(f"Unknown input type: {datasets.__type__}.")
