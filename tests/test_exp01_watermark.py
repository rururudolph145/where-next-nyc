"""Tests for orchestration_experiment/dags/exp01_watermark.py.

PBI EXP-01 (experimentation) — the watermark "now-signal" primitive that the
3-task experiment DAG (read -> advance -> write) delegates to.

The logic lives in a PURE, Airflow-free module so it is unit-testable without
Airflow installed (the experiment container has Airflow; the host test env does
NOT). The DAG file is a thin wrapper; its structure is asserted separately.

Contract (mirrors app/core/watermark for a future drop-in refactor):
  * On-disk format is JSON ``{"position": "<naive-iso>"}``.
  * The watermark is tz-NAIVE throughout (raw ``tpep_pickup_datetime`` is
    naive NYC wall-clock). A tz-aware value is coerced to naive defensively.
  * Writes are atomic (``.tmp`` + ``os.replace``) so a crash mid-write never
    corrupts the file and a failed run leaves the prior watermark intact.

AC coverage (from PBI EXP-01 §7):
  - read_position falls back to seed on first run (missing file)
  - next_window advances by exactly slice_duration
  - write_position round-trips and matches the production JSON format
  - failure leaves the watermark file unchanged (atomic write)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orchestration_experiment.dags.exp01_watermark import (
    next_window,
    read_position,
    write_position,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SEED = datetime(2024, 6, 3, 0, 0, 0)  # naive NYC wall-clock; known busy Monday
SLICE = timedelta(minutes=15)


# ---------------------------------------------------------------------------
# TestReadPosition
# ---------------------------------------------------------------------------


class TestReadPosition:
    """read_position resolves the current simulation 'now' from disk."""

    def test_missing_file_returns_seed(self, tmp_path: Path) -> None:
        """First run (no control file) falls back to the seed. AC EXP-01 §7."""
        # Arrange
        path = tmp_path / ".current_watermark"

        # Act
        result = read_position(path, seed=SEED)

        # Assert
        assert result == SEED

    def test_existing_file_returns_stored_position(self, tmp_path: Path) -> None:
        """A present control file wins over the seed."""
        # Arrange
        path = tmp_path / ".current_watermark"
        stored = datetime(2024, 6, 3, 8, 30, 0)
        path.write_text(json.dumps({"position": stored.isoformat()}), encoding="utf-8")

        # Act
        result = read_position(path, seed=SEED)

        # Assert
        assert result == stored

    def test_returned_position_is_naive(self, tmp_path: Path) -> None:
        """A tz-aware stored value is coerced to naive (contract is naive)."""
        # Arrange
        path = tmp_path / ".current_watermark"
        aware = datetime(2024, 6, 3, 8, 30, 0, tzinfo=timezone.utc)
        path.write_text(json.dumps({"position": aware.isoformat()}), encoding="utf-8")

        # Act
        result = read_position(path, seed=SEED)

        # Assert
        assert result.tzinfo is None

    def test_malformed_file_raises_value_error(self, tmp_path: Path) -> None:
        """A corrupt control file is a hard error, not a silent seed reset."""
        # Arrange
        path = tmp_path / ".current_watermark"
        path.write_text("not json at all", encoding="utf-8")

        # Act / Assert
        with pytest.raises(ValueError):
            read_position(path, seed=SEED)


# ---------------------------------------------------------------------------
# TestNextWindow
# ---------------------------------------------------------------------------


class TestNextWindow:
    """next_window is a pure function: (W, W + slice)."""

    def test_advances_by_exactly_slice(self) -> None:
        """The window end is the position plus one slice_duration."""
        # Arrange
        position = SEED

        # Act
        start, end = next_window(position, slice_=SLICE)

        # Assert
        assert start == SEED
        assert end == datetime(2024, 6, 3, 0, 15, 0)

    def test_start_equals_input_position(self) -> None:
        """The window start is the current position, unchanged."""
        # Arrange
        position = datetime(2024, 6, 3, 8, 30, 0)

        # Act
        start, _ = next_window(position, slice_=SLICE)

        # Assert
        assert start == position

    def test_crosses_hour_boundary(self) -> None:
        """Advancing at :45 rolls into the next hour (boundary value)."""
        # Arrange
        position = datetime(2024, 6, 3, 8, 45, 0)

        # Act
        _, end = next_window(position, slice_=SLICE)

        # Assert
        assert end == datetime(2024, 6, 3, 9, 0, 0)


# ---------------------------------------------------------------------------
# TestWritePosition
# ---------------------------------------------------------------------------


class TestWritePosition:
    """write_position atomically persists the advanced watermark."""

    def test_round_trips_via_read_position(self, tmp_path: Path) -> None:
        """A written position reads back identically."""
        # Arrange
        path = tmp_path / ".current_watermark"
        new_pos = datetime(2024, 6, 3, 0, 15, 0)

        # Act
        write_position(path, new_pos)
        result = read_position(path, seed=SEED)

        # Assert
        assert result == new_pos

    def test_matches_production_json_format(self, tmp_path: Path) -> None:
        """On-disk format is {"position": "<iso>"} — matches app.core.watermark."""
        # Arrange
        path = tmp_path / ".current_watermark"
        new_pos = datetime(2024, 6, 3, 0, 15, 0)

        # Act
        write_position(path, new_pos)
        payload = json.loads(path.read_text(encoding="utf-8"))

        # Assert
        assert payload == {"position": new_pos.isoformat()}

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Writing into a not-yet-existing dir creates parents (first run)."""
        # Arrange
        path = tmp_path / "nested" / "dir" / ".current_watermark"
        new_pos = datetime(2024, 6, 3, 0, 15, 0)

        # Act
        write_position(path, new_pos)

        # Assert
        assert path.is_file()

    def test_tz_aware_value_is_coerced_naive(self, tmp_path: Path) -> None:
        """A tz-aware position is stripped to naive before persisting."""
        # Arrange
        path = tmp_path / ".current_watermark"
        aware = datetime(2024, 6, 3, 0, 15, 0, tzinfo=timezone.utc)

        # Act
        write_position(path, aware)
        payload = json.loads(path.read_text(encoding="utf-8"))

        # Assert
        assert "+00:00" not in payload["position"]
        assert payload["position"] == datetime(2024, 6, 3, 0, 15, 0).isoformat()

    def test_leaves_no_tmp_file_behind(self, tmp_path: Path) -> None:
        """The atomic .tmp sibling is renamed away, not left on disk."""
        # Arrange
        path = tmp_path / ".current_watermark"

        # Act
        write_position(path, SEED)

        # Assert
        leftover = list(tmp_path.glob("*.tmp"))
        assert leftover == []

    def test_failed_write_leaves_prior_watermark_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Crash safety (AC EXP-01 §7): a mid-write failure preserves the old value.

        Mirrors the real container failure observed on 2026-07-14 (UID mismatch
        made the write fail): the watermark must NOT advance so the next run
        reprocesses the same slice.
        """
        # Arrange: a valid prior watermark on disk
        path = tmp_path / ".current_watermark"
        write_position(path, SEED)
        # Make the atomic rename blow up, simulating a crash / permission error
        import orchestration_experiment.dags.exp01_watermark as mod

        def boom(*_args: object, **_kwargs: object) -> None:
            raise OSError("simulated write failure")

        monkeypatch.setattr(mod.os, "replace", boom)

        # Act
        with pytest.raises(OSError):
            write_position(path, datetime(2024, 6, 3, 0, 15, 0))

        # Assert: the prior watermark is untouched
        assert read_position(path, seed=SEED) == SEED
