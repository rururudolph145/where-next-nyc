"""Pipeline-level unit tests for the PBI 25b FIX — watermark-advancing
incremental pipeline.

Covers the two new ``app.core.watermark`` data-watermark functions
(``read_watermark`` / ``write_watermark``) and the windowed ingest path on
``app.ingestion.ingest.Ingestion.run``.

Story ACs exercised here:
  * read_watermark returns None when the control file is absent.
  * read/write roundtrip preserves a NAIVE datetime.
  * write is atomic (no leftover ``.tmp``) and stores no ``Z`` suffix.
  * malformed control file raises ValueError.
  * Ingestion.run filters to ``[window_start, window_end)`` (naive bounds).
  * Ingestion.run with no window args is backward compatible (full range).
  * slice_duration != feature.bucket raises AssertionError in __init__.
  * a zero-trip slice still advances the watermark (empty bronze, no crash).

Test strategy: pure filesystem + in-memory pandas; no Airflow runtime, no
external I/O. AAA with explicit comments; class grouping; AC-tagged
docstrings.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.core.watermark import (
    _WATERMARK_FILE,
    read_watermark,
    write_watermark,
)
from app.ingestion.ingest import Ingestion

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bronze_schema() -> dict[str, str]:
    """Minimal bronze schema sufficient for the windowed-ingest fixture."""
    return {
        "VendorID": "float64",
        "tpep_pickup_datetime": "datetime64[us]",
        "tpep_dropoff_datetime": "datetime64[us]",
        "passenger_count": "float64",
        "trip_distance": "float64",
        "PULocationID": "float64",
        "DOLocationID": "float64",
        "fare_amount": "float64",
    }


def _write_daily_parquet(path: Path, start: pd.Timestamp, n_rows: int) -> None:
    """Write a tiny daily parquet with 15-min-spaced naive pickup times."""
    timestamps = pd.date_range(start, periods=n_rows, freq="15min")
    df = pd.DataFrame(
        {
            "VendorID": [1.0] * n_rows,
            "tpep_pickup_datetime": timestamps,
            "tpep_dropoff_datetime": timestamps + pd.Timedelta(minutes=10),
            "passenger_count": [1.0] * n_rows,
            "trip_distance": [1.5] * n_rows,
            "PULocationID": [100.0] * n_rows,
            "DOLocationID": [200.0] * n_rows,
            "fare_amount": [10.0] * n_rows,
        }
    )
    df.to_parquet(path, index=False)


@pytest.fixture
def two_bucket_raw_tree(tmp_path: Path) -> Path:
    """Raw tree spanning two 15-min buckets on 2021-01-18.

    Pickups at 00:00, 00:15, 00:30, 00:45 — so a [00:00, 00:15) window
    captures exactly one row and a [00:00, 00:30) window captures two.
    """
    raw_root = tmp_path / "raw"
    month_dir = raw_root / "2021_01"
    month_dir.mkdir(parents=True, exist_ok=True)
    _write_daily_parquet(
        month_dir / "2021_01_18.parquet",
        start=pd.Timestamp("2021-01-18 00:00:00"),
        n_rows=4,
    )
    return raw_root


def _ingest_config(raw_root: Path, bronze_path: Path) -> dict[str, Any]:
    """Build an Ingestion config pointing at the temp raw / bronze paths."""
    return {
        "data": {
            "raw_path": str(raw_root),
            "bronze_path": str(bronze_path),
        },
        "ingest": {
            "start_year_month": "2021-01",
            "end_year_month": "2021-01",
            "bronze_schema": _bronze_schema(),
        },
    }


# ---------------------------------------------------------------------------
# TestReadWatermark
# ---------------------------------------------------------------------------


class TestReadWatermark:
    """``read_watermark`` — absent file, roundtrip, malformed control file."""

    def test_read_watermark_returns_none_when_file_absent(self, tmp_path: Path) -> None:
        """read_watermark returns None when no control file exists yet."""
        # Arrange
        root = tmp_path / "wm"

        # Act
        result = read_watermark(root)

        # Assert
        assert result is None, (
            "read_watermark must return None on the first run (control file "
            "absent) so the DAG falls back to ingest.seed_start."
        )

    def test_read_watermark_returns_naive_datetime_after_write(
        self, tmp_path: Path
    ) -> None:
        """read/write roundtrip preserves a NAIVE datetime value."""
        # Arrange
        root = tmp_path / "wm"
        position = datetime(2021, 1, 18, 0, 0, 0)
        write_watermark(root, position)

        # Act
        result = read_watermark(root)

        # Assert
        assert result == position
        assert result is not None and result.tzinfo is None, (
            "read_watermark must return a tz-naive datetime; the data timeline "
            "is tz-naive NYC wall-clock."
        )

    def test_read_watermark_raises_value_error_on_malformed_json(
        self, tmp_path: Path
    ) -> None:
        """A malformed control file raises ValueError (not a silent None)."""
        # Arrange
        root = tmp_path / "wm"
        root.mkdir(parents=True, exist_ok=True)
        (root / _WATERMARK_FILE).write_text("{not valid json", encoding="utf-8")

        # Act / Assert
        with pytest.raises(ValueError, match="Malformed watermark"):
            read_watermark(root)

    def test_read_watermark_raises_value_error_on_missing_position_key(
        self, tmp_path: Path
    ) -> None:
        """A control file lacking the 'position' key raises ValueError."""
        # Arrange
        root = tmp_path / "wm"
        root.mkdir(parents=True, exist_ok=True)
        (root / _WATERMARK_FILE).write_text('{"pos": "2021-01-18"}', encoding="utf-8")

        # Act / Assert
        with pytest.raises(ValueError, match="Malformed watermark"):
            read_watermark(root)


# ---------------------------------------------------------------------------
# TestWriteWatermark
# ---------------------------------------------------------------------------


class TestWriteWatermark:
    """``write_watermark`` — atomicity, naive serialisation, tz coercion."""

    def test_write_watermark_leaves_no_tmp_file(self, tmp_path: Path) -> None:
        """Atomic write leaves no leftover ``.tmp`` sibling."""
        # Arrange
        root = tmp_path / "wm"

        # Act
        write_watermark(root, datetime(2021, 1, 18, 0, 0, 0))

        # Assert
        tmp_file = root / (_WATERMARK_FILE + ".tmp")
        assert not tmp_file.exists(), (
            "write_watermark must os.replace the .tmp file onto the final "
            "path, leaving no half-written sibling behind."
        )
        assert (root / _WATERMARK_FILE).is_file()

    def test_write_watermark_stores_naive_iso_without_z(self, tmp_path: Path) -> None:
        """The persisted position is a NAIVE ISO string (no 'Z', no offset)."""
        # Arrange
        root = tmp_path / "wm"

        # Act
        write_watermark(root, datetime(2021, 1, 18, 0, 0, 0))

        # Assert
        content = (root / _WATERMARK_FILE).read_text(encoding="utf-8")
        assert "Z" not in content
        assert "+00:00" not in content
        assert "2021-01-18T00:00:00" in content

    def test_write_watermark_coerces_tz_aware_to_naive(self, tmp_path: Path) -> None:
        """A tz-aware input is coerced to naive (the contract is naive)."""
        # Arrange
        root = tmp_path / "wm"
        aware = datetime(2021, 1, 18, 0, 0, 0, tzinfo=timezone.utc)

        # Act
        write_watermark(root, aware)
        result = read_watermark(root)

        # Assert
        assert result is not None and result.tzinfo is None, (
            "write_watermark must strip tzinfo so the stored watermark stays "
            "naive-comparable with the data timeline."
        )


# ---------------------------------------------------------------------------
# TestIngestionWindow
# ---------------------------------------------------------------------------


class TestIngestionWindow:
    """``Ingestion.run`` windowed-filter behaviour and backward compat."""

    def test_run_filters_to_window_left_inclusive(
        self, tmp_path: Path, two_bucket_raw_tree: Path
    ) -> None:
        """A [00:00, 00:15) window keeps only the 00:00 row (left-inclusive)."""
        # Arrange
        bronze_path = tmp_path / "bronze" / "bronze.parquet"
        config = _ingest_config(two_bucket_raw_tree, bronze_path)

        # Act — one-bucket window
        Ingestion(config=config).run(
            window_start=datetime(2021, 1, 18, 0, 0, 0),
            window_end=datetime(2021, 1, 18, 0, 15, 0),
        )

        # Assert — only the 00:00 pickup falls in [00:00, 00:15)
        bronze = pd.read_parquet(bronze_path)
        assert len(bronze) == 1
        assert bronze["tpep_pickup_datetime"].iloc[0] == pd.Timestamp(
            "2021-01-18 00:00:00"
        )

    def test_run_window_captures_two_buckets(
        self, tmp_path: Path, two_bucket_raw_tree: Path
    ) -> None:
        """A [00:00, 00:30) window keeps the 00:00 and 00:15 rows only."""
        # Arrange
        bronze_path = tmp_path / "bronze" / "bronze.parquet"
        config = _ingest_config(two_bucket_raw_tree, bronze_path)

        # Act
        Ingestion(config=config).run(
            window_start=datetime(2021, 1, 18, 0, 0, 0),
            window_end=datetime(2021, 1, 18, 0, 30, 0),
        )

        # Assert
        bronze = pd.read_parquet(bronze_path)
        assert len(bronze) == 2
        assert bronze["tpep_pickup_datetime"].max() == pd.Timestamp(
            "2021-01-18 00:15:00"
        )

    def test_run_without_window_args_is_backward_compatible(
        self, tmp_path: Path, two_bucket_raw_tree: Path
    ) -> None:
        """run() with no window args ingests the full range unchanged."""
        # Arrange
        bronze_path = tmp_path / "bronze" / "bronze.parquet"
        config = _ingest_config(two_bucket_raw_tree, bronze_path)

        # Act — no window args
        Ingestion(config=config).run()

        # Assert — all 4 rows present (no filtering)
        bronze = pd.read_parquet(bronze_path)
        assert len(bronze) == 4

    def test_run_zero_trip_slice_writes_empty_bronze_without_crash(
        self, tmp_path: Path, two_bucket_raw_tree: Path
    ) -> None:
        """A window with no matching rows writes empty-but-schema-valid bronze.

        This is the edge case backing ``advance_watermark`` on an empty
        slice: the empty df flows through save_bronze and the watermark
        still advances (the DAG calls advance_watermark on all_success).
        """
        # Arrange — window well before any data
        bronze_path = tmp_path / "bronze" / "bronze.parquet"
        config = _ingest_config(two_bucket_raw_tree, bronze_path)

        # Act
        Ingestion(config=config).run(
            window_start=datetime(2020, 1, 1, 0, 0, 0),
            window_end=datetime(2020, 1, 1, 0, 15, 0),
        )

        # Assert — empty frame, schema columns intact, no exception
        bronze = pd.read_parquet(bronze_path)
        assert len(bronze) == 0
        assert "tpep_pickup_datetime" in bronze.columns


# ---------------------------------------------------------------------------
# TestSliceDurationGuard
# ---------------------------------------------------------------------------


class TestSliceDurationGuard:
    """``Ingestion.__init__`` slice_duration == feature.bucket guard."""

    def test_init_raises_when_slice_duration_mismatches_bucket(
        self, tmp_path: Path
    ) -> None:
        """slice_duration != feature.bucket raises AssertionError."""
        # Arrange
        config: dict[str, Any] = {
            "data": {
                "raw_path": str(tmp_path / "raw"),
                "bronze_path": str(tmp_path / "bronze.parquet"),
            },
            "ingest": {
                "start_year_month": "2021-01",
                "end_year_month": "2021-01",
                "bronze_schema": _bronze_schema(),
                "slice_duration": "30min",
            },
            "feature": {"bucket": "15min"},
        }

        # Act / Assert
        with pytest.raises(AssertionError, match="must equal"):
            Ingestion(config=config)

    def test_init_accepts_matching_slice_duration_and_bucket(
        self, tmp_path: Path
    ) -> None:
        """slice_duration == feature.bucket constructs without error."""
        # Arrange
        config: dict[str, Any] = {
            "data": {
                "raw_path": str(tmp_path / "raw"),
                "bronze_path": str(tmp_path / "bronze.parquet"),
            },
            "ingest": {
                "start_year_month": "2021-01",
                "end_year_month": "2021-01",
                "bronze_schema": _bronze_schema(),
                "slice_duration": "15min",
            },
            "feature": {"bucket": "15min"},
        }

        # Act — should not raise
        ingestion = Ingestion(config=config)

        # Assert
        assert ingestion.start_ym == "2021-01"


# ---------------------------------------------------------------------------
# TestNextWindow
# ---------------------------------------------------------------------------


class TestNextWindow:
    """next_window is a pure two-liner: (position, position + duration)."""

    def test_returns_position_and_position_plus_duration(self) -> None:
        """The window is [position, position + duration), naive throughout."""
        # Arrange
        from datetime import timedelta

        from app.core.watermark import next_window

        position = datetime(2021, 1, 18, 0, 0, 0)
        duration = timedelta(minutes=15)

        # Act
        start, end = next_window(position, duration)

        # Assert
        assert start == datetime(2021, 1, 18, 0, 0, 0)
        assert end == datetime(2021, 1, 18, 0, 15, 0)

    def test_zero_duration_returns_position_twice(self) -> None:
        """Edge: a zero-length window is a valid input (returns (p, p))."""
        # Arrange
        from datetime import timedelta

        from app.core.watermark import next_window

        position = datetime(2021, 1, 18, 0, 0, 0)

        # Act
        start, end = next_window(position, timedelta(0))

        # Assert
        assert start == position
        assert end == position

    def test_preserves_naive_tz(self) -> None:
        """Naive-in, naive-out — the ingest window contract is naive."""
        # Arrange
        from datetime import timedelta

        from app.core.watermark import next_window

        position = datetime(2021, 1, 18, 0, 0, 0)

        # Act
        start, end = next_window(position, timedelta(minutes=15))

        # Assert
        assert start.tzinfo is None
        assert end.tzinfo is None
