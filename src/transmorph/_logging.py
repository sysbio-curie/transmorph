#!/usr/bin/env python3

from datetime import datetime

import os
import logging
import random
import string

# Simple function that creates module global logger.

_DEFAULT_LEVEL_CONSOLE = logging.INFO
_DEFAULT_LEVEL_FILE = logging.DEBUG


def create_logger():

    min_level = min(_DEFAULT_LEVEL_CONSOLE, _DEFAULT_LEVEL_FILE)
    # create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(min_level)

    # create formatter
    console_formatter = logging.Formatter(
        "%(message)s",
    )
    file_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
    )

    # create console handler, file handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(_DEFAULT_LEVEL_CONSOLE)
    ch.setFormatter(console_formatter)

    # add ch to logger
    logger.addHandler(ch)

    log_dir_path = f"{os.path.dirname(__file__)}/logs/"
    if not os.path.exists(log_dir_path):
        os.mkdir(log_dir_path)
    randstr = "".join(random.choices(string.ascii_letters, k=10))
    file_name = f"{datetime.now()}_{randstr}.log".replace(" ", "_")
    file_path = f"{log_dir_path}{file_name}"
    if os.path.exists(file_path):
        os.remove(file_path)
    fh = logging.FileHandler(file_path)
    fh.setLevel(_DEFAULT_LEVEL_FILE)
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    logger.debug("Logger initialized.")

    return logger


logger = create_logger()
