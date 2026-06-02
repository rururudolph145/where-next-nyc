"""Tests for ``_SpeedupClock`` in ``orchestration/dags/_clock_factory.py``.

Covers:
- Proportional time advance (~15 sim-min in ~2s wall-clock at 450x)
- Satisfies the ``Clock`` runtime-checkable protocol
- Thread-safe concurrent ``now()`` calls
- Clean shutdown of the background thread
"""

from __future__ import annotations

import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from app.core.clock import Clock
from orchestration.dags._clock_factory import _SpeedupClock

_T0 = datetime(2024, 6, 3, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def speedup_clock() -> Generator[_SpeedupClock]:
    """Create a 450x speedup clock and stop it after the test."""
    clk = _SpeedupClock(start=_T0, speedup=450)
    yield clk
    clk.stop()


class TestSpeedupClock:
    """Unit tests for the _SpeedupClock wrapper."""

    def test_speedup_clock_advances_proportionally_to_wall_time(self) -> None:
        """Drive _SpeedupClock(start=T0, speedup=450) for ~2s wall-time.

        Assert sim-time advanced ~15 min (900s) with +/-10% tolerance to
        absorb scheduler jitter.
        """
        # Arrange
        clk = _SpeedupClock(start=_T0, speedup=450)

        # Act
        time.sleep(2.0)
        sim_now = clk.now()
        clk.stop()

        # Assert
        sim_delta = (sim_now - _T0).total_seconds()
        expected = 2.0 * 450  # 900 seconds = 15 minutes
        tolerance = expected * 0.10  # +/-10%
        assert abs(sim_delta - expected) < tolerance, (
            f"Expected ~{expected}s sim advance, got {sim_delta:.1f}s "
            f"(tolerance ±{tolerance:.0f}s)"
        )

    def test_speedup_clock_satisfies_clock_protocol(
        self, speedup_clock: _SpeedupClock
    ) -> None:
        """_SpeedupClock satisfies the runtime-checkable Clock protocol."""
        # Assert
        assert isinstance(speedup_clock, Clock)

    def test_speedup_clock_now_is_thread_safe(
        self, speedup_clock: _SpeedupClock
    ) -> None:
        """Concurrent now() calls from multiple threads raise no exception."""
        # Arrange
        errors: list[Exception] = []

        def _call_now() -> datetime:
            try:
                return speedup_clock.now()
            except Exception as exc:
                errors.append(exc)
                raise

        # Act
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_call_now) for _ in range(20)]
            results = [f.result() for f in futures]

        # Assert
        assert not errors
        assert len(results) == 20
        assert all(isinstance(r, datetime) for r in results)

    def test_speedup_clock_stop_halts_background_thread(self) -> None:
        """After stop(), the background thread is no longer alive."""
        # Arrange
        clk = _SpeedupClock(start=_T0, speedup=450)
        assert clk._thread.is_alive()

        # Act
        clk.stop()

        # Assert
        assert not clk._thread.is_alive()

    def test_speedup_clock_has_speedup_attribute(
        self, speedup_clock: _SpeedupClock
    ) -> None:
        """The factory test asserts _speedup attribute exists with correct value."""
        # Assert
        assert speedup_clock._speedup == 450
