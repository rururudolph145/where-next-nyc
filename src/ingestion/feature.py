"""Backwards-compatibility shim — re-exports ``FeatureEngineer`` from its new home.

The implementation now lives in :mod:`app.ingestion.feature`. This module exists
only so legacy imports (``from src.ingestion.feature import FeatureEngineer``) keep
working during the ``src/`` → ``app/`` migration. It will be removed once all
callers have been updated.
"""

from app.ingestion.feature import FeatureEngineer

__all__ = ["FeatureEngineer"]
