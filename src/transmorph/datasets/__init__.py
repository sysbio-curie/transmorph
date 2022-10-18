#!/usr/bin/env python3

from .databank_api import check_files
from .datasets import load_bank
from .datasets import remove_bank

from .datasets import load_chen_10x
from .datasets import load_cell_cycle
from .datasets import load_pal_10x
from .datasets import load_spirals
from .datasets import load_test_datasets_random
from .datasets import load_test_datasets_small
from .datasets import load_travaglini_10x
from .datasets import load_zhou_10x

__all__ = [
    "check_files",
    "load_bank",
    "remove_bank",
    "load_spirals",
    "load_test_datasets_random",
    "load_test_datasets_small",
    "load_cell_cycle",
    "load_chen_10x",
    "load_pal_10x",
    "load_travaglini_10x",
    "load_zhou_10x",
]
