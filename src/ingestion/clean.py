"""Backwards-compatibility shim — re-exports ``DataCleaner`` from its new home.

The implementation now lives in :mod:`app.ingestion.clean`. This module exists
only so legacy imports (``from src.ingestion.clean import DataCleaner``) keep
working during the ``src/`` → ``app/`` migration. It will be removed once all
callers have been updated.
"""

from app.ingestion.clean import DataCleaner

__all__ = ["DataCleaner"]
