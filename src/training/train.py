"""Backwards-compatibility shim — re-exports from the canonical training module.

The implementation now lives in :mod:`app.training.trainer`. This module exists
only so legacy imports (``from src.training.train import Trainer``) and legacy
``unittest.mock.patch`` paths (``patch("src.training.train.mlflow")``) keep
working during the ``src/`` → ``app/`` migration. It will be removed once all
callers have been updated.
"""

from app.training.trainer import MODELS, Trainer, mlflow

__all__ = ["MODELS", "Trainer", "mlflow"]
