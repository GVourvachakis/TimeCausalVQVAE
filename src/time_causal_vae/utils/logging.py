"""Logging helpers for the target package."""

from __future__ import annotations

import logging


def get_console_logger(name: str) -> logging.Logger:
    """Create a console logger matching the legacy helper.

    Parameters
    ----------
    name:
        Logger name passed to :func:`logging.getLogger`.

    Returns
    -------
    logging.Logger
        Logger configured at ``INFO`` level with a stream handler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    c_handler = logging.StreamHandler()
    logger.addHandler(c_handler)
    return logger
