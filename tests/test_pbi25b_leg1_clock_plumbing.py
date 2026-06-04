"""Acceptance tests for PBI 25b Leg 1 — clock plumbing end-to-end smoke test.

Scope: Leg 1 only (clock discipline guard + dependency direction invariant).
Legs 2 and 3 (``/health`` extension and the DAG itself) are out of scope.

The central acceptance question for Leg 1 is: **did the refactor actually
wire the Clock protocol through the callsites, or did the builder just write
unit tests and leave the wiring as dead code?**

This file contains one integration-level acceptance test that proves the
plumbing is real:

  * ``write_validation_log()`` in ``app/ingestion/schema.py`` (the callsite
    that gained a new ``clock: Clock | None = None`` keyword argument) is
    called with a ``SimulatedClock`` frozen at a known UTC instant.
  * The resulting JSON log file's ``validated_at`` timestamp must equal the
    simulated instant — NOT the wall clock.
  * This distinguishes "the parameter was added but ignored" from "the
    parameter was added and actually consumed".

AC coverage mapping
--------------------
AC #11 — Clock-discipline regression guard
         ``tests/test_clock_discipline.py::TestClockDiscipline::
           test_no_production_module_uses_banned_time_calls``
         (pre-existing after builder commit 63b10fc; verified by this run).
         No additional test needed — the story says "verify the existing test
         implements the contract correctly."

AC #12 — Dependency-direction invariant
         ``tests/test_pbi25a_orchestrator_skeleton.py::
           TestAppDoesNotImportOrchestration::
           test_app_does_not_import_from_orchestration``
         (pre-existing after PBI 25a; still green after Leg 1 edits).
         No additional test needed — the story says "verify it still passes."

Integration smoke test
-----------------------
``test_write_validation_log_with_simulated_clock_uses_injected_time`` — proves
the Clock injection plumbing in ``write_validation_log()`` is wired correctly.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.core.clock import SimulatedClock
from app.ingestion.schema import write_validation_log


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_SIMULATED_NOW = datetime(2024, 6, 3, 12, 0, 0, tzinfo=timezone.utc)
"""A fixed UTC instant used as the simulated clock's start time.

Chosen to be recognisably different from any wall-clock value
(June 2024) so that a false positive against the wall clock is
implausible when tests are run in 2025+.
"""


def _minimal_df() -> pd.DataFrame:
    """Minimal conforming Bronze-shaped DataFrame for schema.write_validation_log()."""
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.array(
                ["2024-06-03 08:00:00", "2024-06-03 09:00:00"],
                dtype="datetime64[us]",
            ),
            "tpep_dropoff_datetime": pd.array(
                ["2024-06-03 08:30:00", "2024-06-03 09:30:00"],
                dtype="datetime64[us]",
            ),
            "PULocationID": [1.0, 2.0],
            "passenger_count": [1.0, 1.0],
            "trip_distance": [1.5, 2.0],
        }
    )


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------


class TestClockInjectionPlumbing:
    """Integration smoke test: Clock is actually consumed, not just accepted."""

    def test_write_validation_log_with_simulated_clock_uses_injected_time(
        self, tmp_path: Path
    ) -> None:
        """AC integration: write_validation_log() uses the injected clock's time.

        Arrange: a SimulatedClock frozen at _SIMULATED_NOW (June 2024) is
            constructed; a minimal DataFrame and an empty failure_reasons list
            are prepared.
        Act: write_validation_log() is called with the injected clock.
        Assert: the written JSON log's ``validated_at`` field equals the ISO-8601
            representation of _SIMULATED_NOW, NOT the wall-clock time.

        This test would fail if the parameter were silently ignored (i.e. if
        ``write_validation_log()`` called ``WallClock().now()`` instead of
        consuming the injected clock).
        """
        # Arrange
        log_dir = tmp_path / "validation_logs"
        clock = SimulatedClock(start=_SIMULATED_NOW)
        df = _minimal_df()
        failure_reasons: list[str] = []

        # Act
        log_path = write_validation_log(
            log_dir,
            df,
            failure_reasons,
            clock=clock,
        )

        # Assert — log file was created
        assert log_path.exists(), (
            "write_validation_log() must return the path of a file that exists"
        )

        # Assert — JSON is well-formed
        with open(log_path) as fh:
            payload = json.load(fh)

        # Assert — validated_at matches the simulated instant exactly
        expected_validated_at = _SIMULATED_NOW.isoformat().replace("+00:00", "Z")
        assert payload["validated_at"] == expected_validated_at, (
            f"validated_at should be the simulated clock's time "
            f"({expected_validated_at!r}), got {payload['validated_at']!r}. "
            "This indicates the Clock injection was not consumed — "
            "write_validation_log() is using the wall clock instead of the "
            "injected SimulatedClock."
        )

    def test_write_validation_log_without_clock_arg_still_writes_log(
        self, tmp_path: Path
    ) -> None:
        """Regression guard: the default None path (WallClock fallback) still works.

        Proves the ``clock=None`` default path is not broken by the injection
        refactor.  The exact timestamp is wall-clock and cannot be asserted
        deterministically, so we assert structural correctness only.
        """
        # Arrange
        log_dir = tmp_path / "validation_logs_default"
        df = _minimal_df()

        # Act
        log_path = write_validation_log(log_dir, df, [], clock=None)

        # Assert — file written and has required fields
        assert log_path.exists()
        with open(log_path) as fh:
            payload = json.load(fh)

        for key in (
            "validated_at",
            "rows_in",
            "rows_passing",
            "rows_failing",
            "failure_reasons",
        ):
            assert key in payload, f"Expected key {key!r} missing from log"

        # Assert — validated_at is a parseable ISO-8601 string
        ts_str = payload["validated_at"]
        # Should end with 'Z' (the production format)
        assert ts_str.endswith("Z"), (
            f"validated_at should end with 'Z' (UTC indicator), got {ts_str!r}"
        )
        # Should be parseable as a datetime (robust format check)
        parsed = pd.Timestamp(ts_str)
        assert parsed.year >= 2024, f"Parsed timestamp year {parsed.year} looks wrong"
