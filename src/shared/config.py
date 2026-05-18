"""Backwards-compatibility shim — re-exports ``load_config`` from its new home.

The implementation now lives in :mod:`app.core.config`. This module exists
only so legacy imports (``from src.shared.config import load_config``)
keep working during the ``src/`` → ``app/`` migration. It will be removed
once all callers have been updated.
"""

from app.core.config import load_config

__all__ = ["load_config"]
