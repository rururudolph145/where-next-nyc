"""Clock factory for the refresh_15min DAG.

Provides ``make_clock()`` which returns a ``WallClock`` in production or a
``_SpeedupClock`` when ``DEMO_MODE=1``.  The ``_SpeedupClock`` wraps a
``SimulatedClock`` and advances it proportionally to wall-clock time by a
factor of ``speedup`` using a background daemon thread.

Thread-safety: ``SimulatedClock`` is documented as not thread-safe, so
``_SpeedupClock`` guards both ``advance()`` and ``now()`` with a
``threading.Lock``.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timedelta, timezone

from app.core.clock import Clock, SimulatedClock, WallClock, set_active_clock


class _SpeedupClock:
    """Clock that advances a ``SimulatedClock`` at ``speedup`` x wall-clock.

    The background thread ticks at roughly 50 ms intervals and advances the
    simulated clock by ``speedup * wall_delta`` on each tick.  A
    ``threading.Event`` is used instead of ``time.sleep`` so that the daemon
    thread can shut down cleanly when ``stop()`` is called or the process
    exits.

    Args:
        start: UTC-aware datetime to initialise the simulated clock.
        speedup: Wall-clock multiplier (e.g. 450 means 1 wall-second equals
            450 simulated seconds).
    """

    def __init__(self, start: datetime, speedup: float) -> None:
        self._sim = SimulatedClock(start)
        self._speedup: float = speedup
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._last_wall: float = time.monotonic()

        self._thread = threading.Thread(
            target=self._tick_loop, daemon=True, name="speedup-clock"
        )
        self._thread.start()

    # -- Clock protocol -------------------------------------------------------

    def now(self) -> datetime:
        """Return the current simulated UTC-aware datetime."""
        with self._lock:
            return self._sim.now()

    # -- Internal -------------------------------------------------------------

    def _tick_loop(self) -> None:
        """Background loop: advance sim-clock proportionally to wall-clock."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=0.05)  # ~50 ms tick
            wall_now = time.monotonic()
            with self._lock:
                wall_delta = wall_now - self._last_wall
                self._last_wall = wall_now
                sim_delta = timedelta(seconds=wall_delta * self._speedup)
                self._sim.advance(sim_delta)

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it to join."""
        self._stop_event.set()
        self._thread.join(timeout=2.0)


# -- Factory -----------------------------------------------------------------

# Known busy Monday used as the demo start time (per brief section 3b).
_DEMO_START = datetime(2024, 6, 3, 0, 0, tzinfo=timezone.utc)
_DEMO_SPEEDUP = 450


def make_clock() -> Clock:
    """Return a ``Clock`` appropriate for the current execution mode.

    As a side effect, registers the returned clock as the process-global
    active clock via ``set_active_clock``.  This allows ``app/api/main.py``
    to read the active clock via ``get_active_clock()`` without importing
    from ``orchestration/``.

    Returns:
        ``WallClock`` when ``DEMO_MODE`` env var is absent or not ``"1"``.
        ``_SpeedupClock(start=2024-06-03T00:00Z, speedup=450)`` otherwise.
    """
    if os.environ.get("DEMO_MODE") == "1":
        clock: Clock = _SpeedupClock(start=_DEMO_START, speedup=_DEMO_SPEEDUP)
    else:
        clock = WallClock()
    set_active_clock(clock)
    return clock
