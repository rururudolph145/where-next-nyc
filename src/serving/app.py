"""Backwards-compatibility shim — re-exports the FastAPI app from its new home.

The implementation now lives in :mod:`app.api.main`. This module exists only
so legacy imports (``from src.serving.app import app``,
``import src.serving.app as m``) keep working during the ``src/`` → ``app/``
migration. It will be removed once all callers have been updated.

We alias the module object via :mod:`sys.modules` so that mutating attributes
on ``src.serving.app`` (e.g. ``m._predictions_df = ...`` in tests) mutates the
canonical module's globals — the endpoint handlers read from those globals.
"""

import sys

from app.api import main as _canonical

# Replace this shim module with the canonical one so attribute mutations on
# ``src.serving.app`` propagate to ``app.api.main`` and vice versa.
sys.modules[__name__] = _canonical
