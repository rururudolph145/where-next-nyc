"""Logging setup helper.

A single ``setup_logging`` entrypoint so every script/service uses the
same format. Currently extracted verbatim from ``main.py``'s
``logging.basicConfig`` call.
"""

import logging

DEFAULT_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"


def setup_logging(level: int = logging.INFO, fmt: str = DEFAULT_FORMAT) -> None:
    """Configure the root logger.

    Args:
        level: Logging level (e.g. ``logging.INFO``, ``logging.DEBUG``).
        fmt: Log record format string.
    """
    logging.basicConfig(level=level, format=fmt)
