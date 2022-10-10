#!/usr/bin/env python3

from datetime import datetime

import os
import logging
import random
import string
import sys

FORBIDDEN_FILE_CHARS = '{}[]()/<>:"/\\|?*.'

# Simple function that creates module global logger.


def create_logger():

    DEFAULT_LEVEL = logging.DEBUG

    # create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(DEFAULT_LEVEL)

    # create formatter
    console_formatter = logging.Formatter(
        "%(message)s",
    )
    file_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
    )

    # create console handler, file handler and set level to debug
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(console_formatter)

    # add ch to logger
    logger.addHandler(ch)

    log_dir_path = f"{os.path.dirname(__file__)}/logs/"
    if not os.path.exists(log_dir_path):
        os.mkdir(log_dir_path)
    randstr = "".join(random.choices(string.ascii_letters, k=10))
    file_name = f"{datetime.now()}_{randstr}"
    for c in FORBIDDEN_FILE_CHARS:
        file_name = file_name.replace(c, "_")
    file_name = f"{file_name}.log"
    file_path = f"{log_dir_path}{file_name}"
    if os.path.exists(file_path):
        os.remove(file_path)
    fh = logging.FileHandler(file_path)
    fh.setLevel(DEFAULT_LEVEL)
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    logger.debug("Logger initialized.")

    return logger


logger = create_logger()
