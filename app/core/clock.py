"""Time source abstraction for the NYC Taxi Demand Forecasting project.

Provides a ``Clock`` protocol and two concrete implementations:

- ``WallClock`` — production clock; reads the system clock and returns UTC-aware
  ``datetime`` objects.
- ``SimulatedClock`` — deterministic clock for tests and DAG task unit tests;
  advances only when ``.advance()`` is called explicitly.

Thread-safety note: ``SimulatedClock`` is NOT thread-safe. Callers requiring
concurrent advance from multiple threads must add their own locking around
``.advance()`` and ``.now()`` calls.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Time source abstraction; allows deterministic time control in tests."""

    def now(self) -> datetime:
        """Return the current UTC-aware datetime."""
        ...


class WallClock:
    """Production clock; delegates to the system clock, always UTC-aware."""

    def now(self) -> datetime:
        """Return ``datetime.now(tz=timezone.utc)``."""
        return datetime.now(tz=timezone.utc)


class SimulatedClock:
    """Deterministic clock for tests and DAG task unit tests.

    The clock is frozen at ``start`` until ``.advance()`` is called. There is
    no automatic advance or speedup — every time step is explicit.

    Args:
        start: UTC-aware ``datetime``. Raises ``TypeError`` if naive (i.e.
            ``start.tzinfo is None``). Prevents naive-vs-aware comparison bugs
            when mixing ``SimulatedClock`` and ``WallClock`` output.

    Raises:
        TypeError: if ``start`` has no timezone info (naive datetime).
    """

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise TypeError(
                "SimulatedClock requires a timezone-aware datetime; "
                f"got naive datetime {start!r}. "
                "Pass tzinfo=timezone.utc or use datetime.now(tz=timezone.utc)."
            )
        self._current: datetime = start

    def now(self) -> datetime:
        """Return the current simulated UTC-aware datetime."""
        return self._current

    def advance(self, delta: timedelta) -> None:
        """Move the clock forward by ``delta``.

        Args:
            delta: Amount of time to advance. Must be non-negative.

        Raises:
            ValueError: if ``delta`` is negative. Time does not go backward in
                simulation; explicit rejection at the API boundary catches
                mistakes early.
        """
        if delta < timedelta(0):
            raise ValueError(
                f"SimulatedClock.advance() requires a non-negative delta; "
                f"got {delta!r}. Time does not go backward."
            )
        self._current += delta
