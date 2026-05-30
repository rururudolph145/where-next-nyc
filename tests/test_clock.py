"""Tests for app/core/clock.py — Clock protocol, WallClock, SimulatedClock.

AC coverage:
  AC #2  — WallClock.now() returns UTC-aware datetime within 1 s of system clock
  AC #3  — SimulatedClock.now() returns start; advance() moves it forward
  AC #4  — Both implementations satisfy isinstance(obj, Clock) at runtime
  AC #5  — SimulatedClock.advance(negative) raises ValueError
  AC #6  — SimulatedClock(naive datetime) raises TypeError
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.core.clock import Clock, SimulatedClock, WallClock

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

T0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# TestWallClock
# ---------------------------------------------------------------------------


class TestWallClock:
    """Happy-path tests for WallClock."""

    def test_now_returns_utc_aware_datetime(self) -> None:
        """WallClock.now() must return a datetime with tzinfo == timezone.utc.

        AC #2.
        """
        # Arrange
        clock = WallClock()

        # Act
        result = clock.now()

        # Assert
        assert result.tzinfo is not None
        assert result.tzinfo == timezone.utc

    def test_now_is_within_one_second_of_system_clock(self) -> None:
        """WallClock.now() must be within 1 s of datetime.now(tz=timezone.utc).

        AC #2.
        """
        # Arrange
        clock = WallClock()

        # Act
        result = clock.now()
        system_now = datetime.now(tz=timezone.utc)

        # Assert
        assert abs((result - system_now).total_seconds()) < 1.0


# ---------------------------------------------------------------------------
# TestSimulatedClock
# ---------------------------------------------------------------------------


class TestSimulatedClock:
    """Tests for SimulatedClock — happy path, edge cases, and failure cases."""

    def test_init_with_utc_aware_datetime_succeeds(self) -> None:
        """SimulatedClock(start=aware_dt) must not raise.

        AC #3.
        """
        # Arrange / Act / Assert
        SimulatedClock(start=T0)  # should not raise

    def test_now_returns_start_before_advance(self) -> None:
        """SimulatedClock.now() must equal start before any advance call.

        AC #3.
        """
        # Arrange
        clock = SimulatedClock(start=T0)

        # Act
        result = clock.now()

        # Assert
        assert result == T0

    def test_advance_moves_now_forward_by_delta(self) -> None:
        """After advance(15 min), now() must equal start + 15 min.

        AC #3.
        """
        # Arrange
        clock = SimulatedClock(start=T0)
        delta = timedelta(minutes=15)

        # Act
        clock.advance(delta)

        # Assert
        assert clock.now() == T0 + delta

    def test_advance_with_zero_delta_keeps_now_unchanged(self) -> None:
        """advance(timedelta(0)) must leave now() equal to start.

        Edge case — zero boundary.
        """
        # Arrange
        clock = SimulatedClock(start=T0)

        # Act
        clock.advance(timedelta(0))

        # Assert
        assert clock.now() == T0

    def test_multiple_advances_accumulate(self) -> None:
        """Three consecutive advance(15 min) calls must add up to 45 min total.

        AC #3.
        """
        # Arrange
        clock = SimulatedClock(start=T0)

        # Act
        clock.advance(timedelta(minutes=15))
        clock.advance(timedelta(minutes=15))
        clock.advance(timedelta(minutes=15))

        # Assert
        assert clock.now() == T0 + timedelta(minutes=45)

    def test_init_with_naive_datetime_raises_typeerror(self) -> None:
        """SimulatedClock(start=naive_dt) must raise TypeError.

        AC #6 / Edge Case.
        """
        # Arrange
        naive_dt = datetime(2024, 1, 1, 0, 0, 0)  # no tzinfo

        # Act / Assert
        with pytest.raises(TypeError):
            SimulatedClock(start=naive_dt)

    @pytest.mark.parametrize(
        "delta",
        [
            timedelta(minutes=-1),
            timedelta(seconds=-1),
            timedelta(days=-1),
        ],
    )
    def test_advance_with_negative_delta_raises_valueerror(
        self, delta: timedelta
    ) -> None:
        """advance(negative_delta) must raise ValueError.

        AC #5 / Edge Case.
        """
        # Arrange
        clock = SimulatedClock(start=T0)

        # Act / Assert
        with pytest.raises(ValueError):
            clock.advance(delta)


# ---------------------------------------------------------------------------
# TestProtocolConformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Verify that both implementations satisfy the Clock protocol at runtime."""

    def test_wallclock_satisfies_clock_protocol(self) -> None:
        """isinstance(WallClock(), Clock) must be True.

        AC #4.
        """
        # Arrange
        obj = WallClock()

        # Act / Assert
        assert isinstance(obj, Clock)

    def test_simulatedclock_satisfies_clock_protocol(self) -> None:
        """isinstance(SimulatedClock(start=t0), Clock) must be True.

        AC #4.
        """
        # Arrange
        obj = SimulatedClock(start=T0)

        # Act / Assert
        assert isinstance(obj, Clock)
