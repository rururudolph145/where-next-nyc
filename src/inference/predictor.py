"""Backwards-compatibility shim — re-exports ``Predictor`` from its new home.

The implementation now lives in :mod:`app.inference.predictor`. This module
exists only so legacy imports (``from src.inference.predictor import Predictor``)
keep working during the ``src/`` → ``app/`` migration. It will be removed once
all callers have been updated.
"""

from app.inference.predictor import Predictor

__all__ = ["Predictor"]
