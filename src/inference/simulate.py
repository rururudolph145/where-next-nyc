"""Backwards-compatibility shim — re-exports ``HourlySimulator`` from its new home.

The implementation now lives in :mod:`app.inference.simulator`. This module
exists only so legacy imports
(``from src.inference.simulate import HourlySimulator``) keep working during
the ``src/`` → ``app/`` migration. It will be removed once all callers have
been updated.
"""

from app.inference.simulator import HourlySimulator

__all__ = ["HourlySimulator"]
