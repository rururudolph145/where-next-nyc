"""Tests for ``make_clock()`` factory in ``orchestration/dags/_clock_factory.py``.

Covers:
- Returns WallClock when DEMO_MODE is unset
- Returns _SpeedupClock with speedup=450 when DEMO_MODE=1
- Side-effect: registers the returned clock via set_active_clock()
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.core.clock import WallClock, get_active_clock, set_active_clock
from orchestration.dags._clock_factory import _SpeedupClock, make_clock


@pytest.fixture(autouse=True)
def _reset_active_clock() -> Iterator[None]:
    """Reset the active clock to WallClock() after each test.

    The make_clock() factory has a side effect: it registers its returned
    clock as the active clock. Without this fixture, test order would
    leak state across tests.
    """
    yield
    set_active_clock(WallClock())


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


class TestMakeClockRegistersActiveClock:
    """make_clock() must register its returned clock as the active clock.

    This side effect is how the ``/health`` endpoint sees the simulated time
    without ``app/api/main.py`` importing from ``orchestration/`` — preserving
    the AC #12 dependency-direction invariant.
    """

    def test_make_clock_registers_wallclock_when_demo_mode_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After make_clock() with DEMO_MODE unset, get_active_clock() returns the same WallClock."""
        # Arrange
        monkeypatch.delenv("DEMO_MODE", raising=False)

        # Act
        clock = make_clock()

        # Assert
        assert get_active_clock() is clock
        assert isinstance(get_active_clock(), WallClock)

    def test_make_clock_registers_speedup_clock_when_demo_mode_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After make_clock() with DEMO_MODE=1, get_active_clock() returns the same _SpeedupClock."""
        # Arrange
        monkeypatch.setenv("DEMO_MODE", "1")

        # Act
        clock = make_clock()
        assert isinstance(clock, _SpeedupClock)  # narrow type for mypy + .stop()

        # Assert
        try:
            assert get_active_clock() is clock
        finally:
            clock.stop()  # clean up background thread
