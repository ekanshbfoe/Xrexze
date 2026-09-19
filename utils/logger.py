"""Xrexze Logger — Rotating file + console logger."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(console)

    file_handler = RotatingFileHandler(
        _LOG_DIR / "xrexze.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | "
        "%(funcName)s:%(lineno)d | %(message)s",
    ))
    logger.addHandler(file_handler)

    return logger
