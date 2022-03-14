import os
import logging


def create_logger(fileName: str = None):
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter(
        "%(levelname)s - %(message)s",
    )

    # create console handler, file handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    if fileName:
        if os.path.exists(f"{fileName}.log"):
            os.remove(f"{fileName}.log")
        fh = logging.FileHandler(f"{fileName}.log")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
