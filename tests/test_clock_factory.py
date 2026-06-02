"""Tests for ``make_clock()`` factory in ``orchestration/dags/_clock_factory.py``.

Covers:
- Returns WallClock when DEMO_MODE is unset
- Returns _SpeedupClock with speedup=450 when DEMO_MODE=1
"""

from __future__ import annotations

import pytest

from app.core.clock import WallClock
from orchestration.dags._clock_factory import _SpeedupClock, make_clock


class TestMakeClock:
    """Tests for the make_clock() factory function."""

    def test_make_clock_returns_wallclock_when_demo_mode_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When DEMO_MODE is absent, make_clock() returns a WallClock."""
        # Arrange
        monkeypatch.delenv("DEMO_MODE", raising=False)

        # Act
        clock = make_clock()

        # Assert
        assert isinstance(clock, WallClock)

    def test_make_clock_returns_speedup_clock_when_demo_mode_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When DEMO_MODE=1, make_clock() returns a _SpeedupClock(speedup=450)."""
        # Arrange
        monkeypatch.setenv("DEMO_MODE", "1")

        # Act
        clock = make_clock()

        # Assert
        assert isinstance(clock, _SpeedupClock)
        assert clock._speedup == 450
        clock.stop()  # clean up background thread

    def test_make_clock_returns_wallclock_when_demo_mode_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When DEMO_MODE=0, make_clock() returns a WallClock (not speedup)."""
        # Arrange
        monkeypatch.setenv("DEMO_MODE", "0")

        # Act
        clock = make_clock()

        # Assert
        assert isinstance(clock, WallClock)
