"""
bad_channel_rejection/logging_config.py

Centralised logging setup for the BCR pipeline.

Usage:
    from bad_channel_rejection.logging_config import setup_logging
    logger = setup_logging(__name__)
    logger.info("Loading data...")

Design
------
- Single entry point: setup_logging(name) returns a configured Logger.
- Output format includes timestamp, module name, level, and message.
- Level controlled by BCR_LOG_LEVEL env var (default: INFO).
- File handler writes to logs/bcr_{YYYY-MM-DD}.log when BCR_LOG_TO_FILE=1.
- Idempotent: repeated calls do not duplicate handlers.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(name: str, level: str | None = None) -> logging.Logger:
    """Return a configured logger. Idempotent across calls with same name.

    Parameters
    ----------
    name : str
        Logger name, typically __name__ of the calling module.
    level : str, optional
        Override log level. If None, reads BCR_LOG_LEVEL env var (default INFO).

    Returns
    -------
    logging.Logger
        Logger with stream handler (and optional file handler) attached.
    """
    logger = logging.getLogger(name)

    if getattr(logger, "_bcr_configured", False):
        return logger

    resolved_level = (level or os.environ.get("BCR_LOG_LEVEL", "INFO")).upper()
    logger.setLevel(resolved_level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if os.environ.get("BCR_LOG_TO_FILE") == "1":
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"bcr_{datetime.now().strftime('%Y-%m-%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    logger._bcr_configured = True
    return logger
