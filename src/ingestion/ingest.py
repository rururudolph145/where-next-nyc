"""Backwards-compatibility shim — re-exports ``Ingestion`` from its new home.

The implementation now lives in :mod:`app.ingestion.ingest`. This module exists
only so legacy imports (``from src.ingestion.ingest import Ingestion``) keep
working during the ``src/`` → ``app/`` migration. It will be removed once all
callers have been updated.
"""

from app.ingestion.ingest import Ingestion

__all__ = ["Ingestion"]
