"""Acceptance tests for PBI 19 — Bronze schema-gate end-to-end.

Story: As a pipeline operator, I want Bronze data validated against a defined
schema before it enters the Silver layer, so that downstream cleaning and
feature engineering never silently process corrupt or structurally wrong data.

These tests exercise DataCleaner.run() through on-disk parquet round-trips
(real Bronze parquet → real tmp Silver dir), one test per acceptance criterion.
They sit above the unit tests in tests/test_clean.py::TestSchemaValidation and
prove the story works as a whole observable on disk.

Acceptance criteria covered
----------------------------
AC #1  — conforming Bronze frame → run() completes without exception
AC #2  — missing required column → DataValidationError raised before cleaning
AC #3  — PULocationID outside 1-263 → DataValidationError raised
AC #4  — JSON log written with validated_at, rows_in, rows_passing,
          rows_failing, failure_reasons on every run (pass or fail)
AC #5  — data/silver/_validation/ created automatically if absent
AC #6  — DataValidationError is importable and is not a subclass of ValueError
AC #7  — ruff + mypy gates (tooling AC, not a test — see coverage table)
AC #8  — unit tests in TestSchemaValidation already cover this AC
         (see tests/test_clean.py::TestSchemaValidation)

Edge cases covered
------------------
- 0 rows passing → log still written, exception raised
- datetime64[ns] vs datetime64[us] — both accepted
- extra unexpected columns — not a failure
- PULocationID = 1.0 (float repr of int 1) — accepted
- repeat runs — no silent overwrite (two distinct log files)
- empty Bronze frame — passes schema
- log written even when validation fails (try/finally semantics)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.ingestion.clean import DataCleaner
from app.ingestion.schema import (
    DataValidationError,
    build_bronze_schema,
    validate_bronze_schema,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_bronze_df() -> pd.DataFrame:
    """Minimal conforming Bronze DataFrame — all required columns, valid values."""
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.array(
                [
                    "2023-01-01 08:00:00",
                    "2023-01-01 09:00:00",
                    "2023-01-01 10:00:00",
                ],
                dtype="datetime64[us]",
            ),
            "tpep_dropoff_datetime": pd.array(
                [
                    "2023-01-01 08:15:00",
                    "2023-01-01 09:20:00",
                    "2023-01-01 10:30:00",
                ],
                dtype="datetime64[us]",
            ),
            "PULocationID": pd.array([1.0, 100.0, 263.0], dtype="float64"),
            "passenger_count": pd.array([2.0, 1.0, 3.0], dtype="float64"),
            "trip_distance": pd.array([1.5, 3.2, 2.0], dtype="float64"),
            "fare_amount": pd.array([7.5, 12.0, 9.0], dtype="float64"),
        }
    )


def _make_cleaner(tmp_path: Path, *, bronze_df: pd.DataFrame) -> DataCleaner:
    """Write *bronze_df* to a parquet file and return a configured DataCleaner.

    All paths are rooted inside *tmp_path* so tests are fully hermetic.
    The silver dir is NOT pre-created; tests that exercise AC #5 rely on
    DataCleaner (via write_validation_log) to create it.
    """
    bronze_path = tmp_path / "bronze" / "raw.parquet"
    bronze_path.parent.mkdir(parents=True, exist_ok=True)
    bronze_df.to_parquet(bronze_path, index=False)

    silver_path = tmp_path / "silver" / "cleaned.parquet"
    # Deliberately do NOT create silver_path.parent — AC #5 asserts auto-creation.

    config: dict[str, Any] = {
        "env": "local",
        "data": {
            "bronze_path": str(bronze_path),
            "silver_path": str(silver_path),
        },
        "clean": {"null_strategy": "drop"},
        "inference": {"simulation": {"start_hour": "2023-06-01 00:00:00"}},
        "model": {"store_path": str(tmp_path / "models")},
    }
    return DataCleaner(config)


def _validation_dir(tmp_path: Path) -> Path:
    """Return the expected _validation directory path for a standard tmp layout."""
    return tmp_path / "silver" / "_validation"


def _read_single_log(log_dir: Path) -> dict[str, Any]:
    """Assert exactly one log file exists in *log_dir* and return its contents."""
    log_files = list(log_dir.glob("validation_*.json"))
    assert len(log_files) == 1, (
        f"Expected 1 validation log, found {len(log_files)} in {log_dir}"
    )
    with open(log_files[0]) as fh:
        return json.load(fh)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# AC #1 — Conforming Bronze frame runs to completion
# ---------------------------------------------------------------------------


def test_ac1_conforming_bronze_runs_to_completion(tmp_path: Path) -> None:
    """AC #1: DataCleaner.run() completes without exception on a valid frame."""
    # Arrange
    cleaner = _make_cleaner(tmp_path, bronze_df=_valid_bronze_df())

    # Act / Assert — no exception raised
    cleaner.run()


# ---------------------------------------------------------------------------
# AC #2 — Missing required column raises DataValidationError before cleaning
# ---------------------------------------------------------------------------


def test_ac2_missing_required_column_raises_before_cleaning(tmp_path: Path) -> None:
    """AC #2: Missing required column raises DataValidationError before any cleaning.

    We verify "before cleaning" by checking that the Silver parquet file was
    NOT written — if cleaning had run to completion the Silver file would exist.
    """
    # Arrange — drop PULocationID (a required column)
    bad_df = _valid_bronze_df().drop(columns=["PULocationID"])
    cleaner = _make_cleaner(tmp_path, bronze_df=bad_df)
    silver_path = Path(cleaner.silver_path)

    # Act
    with pytest.raises(DataValidationError):
        cleaner.run()

    # Assert — Silver layer was NOT written (validation halted before cleaning)
    assert not silver_path.exists(), (
        "Silver parquet must not exist when validation fails before cleaning"
    )


@pytest.mark.parametrize(
    "col",
    [
        pytest.param("tpep_pickup_datetime", id="tpep_pickup_datetime"),
        pytest.param("passenger_count", id="passenger_count"),
        pytest.param("trip_distance", id="trip_distance"),
        pytest.param("PULocationID", id="PULocationID"),
    ],
)
def test_ac2_each_required_column_missing_raises(col: str, tmp_path: Path) -> None:
    """AC #2 (parametrised): each sentinel column, when missing, triggers DataValidationError."""
    # Arrange — unique subdir per column so parquet paths never collide
    col_tmp = tmp_path / col
    col_tmp.mkdir()
    bad_df = _valid_bronze_df().drop(columns=[col])
    cleaner = _make_cleaner(col_tmp, bronze_df=bad_df)

    # Act / Assert
    with pytest.raises(DataValidationError):
        cleaner.run()


# ---------------------------------------------------------------------------
# AC #3 — PULocationID outside 1–265 raises DataValidationError
# ---------------------------------------------------------------------------


def test_ac3_pulocationid_out_of_range_raises(tmp_path: Path) -> None:
    """AC #3: PULocationID = 266 (above highest TLC zone) raises DataValidationError."""
    # Arrange
    bad_df = _valid_bronze_df().copy()
    bad_df.loc[0, "PULocationID"] = 266.0
    cleaner = _make_cleaner(tmp_path, bronze_df=bad_df)

    # Act / Assert
    with pytest.raises(DataValidationError):
        cleaner.run()


def test_ac3_pulocationid_zero_raises(tmp_path: Path) -> None:
    """AC #3 (boundary): PULocationID = 0 is below the inclusive lower bound."""
    # Arrange
    bad_df = _valid_bronze_df().copy()
    bad_df.loc[0, "PULocationID"] = 0.0
    cleaner = _make_cleaner(tmp_path, bronze_df=bad_df)

    # Act / Assert
    with pytest.raises(DataValidationError):
        cleaner.run()


# ---------------------------------------------------------------------------
# AC #4 — JSON log written with all five required fields
# ---------------------------------------------------------------------------


def test_ac4_log_written_with_all_required_fields_on_success(tmp_path: Path) -> None:
    """AC #4: On a clean pass, log contains validated_at, rows_in, rows_passing,
    rows_failing, failure_reasons."""
    # Arrange
    bronze_df = _valid_bronze_df()
    cleaner = _make_cleaner(tmp_path, bronze_df=bronze_df)

    # Act
    cleaner.run()

    # Assert — log file written with all required keys
    log_dir = _validation_dir(tmp_path)
    payload = _read_single_log(log_dir)

    required_keys = {
        "validated_at",
        "rows_in",
        "rows_passing",
        "rows_failing",
        "failure_reasons",
    }
    assert required_keys.issubset(payload.keys()), (
        f"Log missing keys: {required_keys - payload.keys()}"
    )


def test_ac4_log_values_correct_on_success(tmp_path: Path) -> None:
    """AC #4: On a clean pass the row counts and failure_reasons list are correct."""
    # Arrange
    bronze_df = _valid_bronze_df()  # 3 rows
    cleaner = _make_cleaner(tmp_path, bronze_df=bronze_df)

    # Act
    cleaner.run()

    # Assert
    log_dir = _validation_dir(tmp_path)
    payload = _read_single_log(log_dir)

    assert payload["rows_in"] == 3
    assert payload["rows_passing"] == 3
    assert payload["rows_failing"] == 0
    assert payload["failure_reasons"] == []
    # validated_at must be parseable ISO-8601 ending with Z
    assert payload["validated_at"].endswith("Z")
    pd.Timestamp(payload["validated_at"])  # raises if not parseable


def test_ac4_log_written_on_failure_with_failure_reasons(tmp_path: Path) -> None:
    """AC #4: On validation failure the log records non-empty failure_reasons."""
    # Arrange
    bad_df = _valid_bronze_df().drop(columns=["PULocationID"])
    cleaner = _make_cleaner(tmp_path, bronze_df=bad_df)

    # Act
    with pytest.raises(DataValidationError):
        cleaner.run()

    # Assert — log written and failure_reasons is non-empty
    log_dir = _validation_dir(tmp_path)
    payload = _read_single_log(log_dir)

    assert payload["failure_reasons"], (
        "failure_reasons must be non-empty when validation fails"
    )
    assert isinstance(payload["failure_reasons"], list)
    # Structural failure → rows_passing must be 0 and rows_failing must equal rows_in
    assert payload["rows_passing"] == 0, (
        "rows_passing must be 0 on a structural failure"
    )
    assert payload["rows_failing"] == payload["rows_in"], (
        "rows_failing must equal rows_in on a structural failure"
    )
    assert payload["rows_in"] == 3, (
        "rows_in must match the 3-row fixture used in this test"
    )


# ---------------------------------------------------------------------------
# AC #5 — _validation/ directory auto-created if absent
# ---------------------------------------------------------------------------


def test_ac5_validation_dir_created_automatically(tmp_path: Path) -> None:
    """AC #5: data/silver/_validation/ is created automatically when missing."""
    # Arrange — confirm the dir does NOT exist before run()
    cleaner = _make_cleaner(tmp_path, bronze_df=_valid_bronze_df())
    log_dir = _validation_dir(tmp_path)
    assert not log_dir.exists(), "Precondition: _validation dir must not exist yet"

    # Act
    cleaner.run()

    # Assert
    assert log_dir.is_dir(), "_validation dir must be created by run()"


# ---------------------------------------------------------------------------
# AC #6 — DataValidationError is importable and not a subclass of ValueError
# ---------------------------------------------------------------------------


def test_ac6_datavalidation_error_is_importable_and_distinct_from_value_error() -> None:
    """AC #6: DataValidationError can be imported and is NOT a subclass of ValueError."""
    # Import is the assertion itself — if the import at module level did not fail
    # then the class is importable.  We then check the MRO.
    assert issubclass(DataValidationError, Exception), (
        "DataValidationError must be a subclass of Exception"
    )
    assert not issubclass(DataValidationError, ValueError), (
        "DataValidationError must NOT be a subclass of ValueError"
    )


def test_ac6_datavalidation_error_carries_failure_reasons() -> None:
    """AC #6: DataValidationError exposes failure_reasons attribute as a list."""
    # Arrange
    reasons = ["column PULocationID not found", "dtype mismatch"]

    # Act
    exc = DataValidationError(reasons)

    # Assert
    assert exc.failure_reasons == reasons
    assert isinstance(exc.failure_reasons, list)


# ---------------------------------------------------------------------------
# Edge: log written even when validation fails (try/finally semantics)
# ---------------------------------------------------------------------------


def test_edge_log_written_even_when_validation_fails(tmp_path: Path) -> None:
    """Edge: the try/finally in run() ensures the log is written before re-raise."""
    # Arrange — out-of-range PULocationID triggers DataValidationError
    bad_df = _valid_bronze_df().copy()
    bad_df.loc[0, "PULocationID"] = 999.0
    cleaner = _make_cleaner(tmp_path, bronze_df=bad_df)

    # Act
    with pytest.raises(DataValidationError):
        cleaner.run()

    # Assert — log file still exists despite the exception
    log_dir = _validation_dir(tmp_path)
    log_files = list(log_dir.glob("validation_*.json"))
    assert len(log_files) == 1, (
        "Log must be written by the finally block even when validation raises"
    )


# ---------------------------------------------------------------------------
# Edge: 0 rows passing → log still written, exception raised
# ---------------------------------------------------------------------------


def test_edge_zero_rows_passing_log_still_written_and_exception_raised(
    tmp_path: Path,
) -> None:
    """Edge: when every row fails (e.g. all PULocationID out of range),
    log is still written and DataValidationError is raised."""
    # Arrange — all three rows have out-of-range PULocationID
    bad_df = _valid_bronze_df().copy()
    bad_df["PULocationID"] = 999.0  # all rows fail
    cleaner = _make_cleaner(tmp_path, bronze_df=bad_df)

    # Act
    with pytest.raises(DataValidationError):
        cleaner.run()

    # Assert — log written
    log_dir = _validation_dir(tmp_path)
    log_files = list(log_dir.glob("validation_*.json"))
    assert len(log_files) == 1


# ---------------------------------------------------------------------------
# Edge: datetime64[ns] is accepted (alongside datetime64[us])
# ---------------------------------------------------------------------------


def test_edge_datetime_ns_resolution_accepted(tmp_path: Path) -> None:
    """Edge: Bronze frames using datetime64[ns] resolution must pass validation."""
    # Arrange — cast pickup datetime to ns resolution
    df = _valid_bronze_df().copy()
    df["tpep_pickup_datetime"] = df["tpep_pickup_datetime"].astype("datetime64[ns]")
    cleaner = _make_cleaner(tmp_path, bronze_df=df)

    # Act / Assert — no exception
    cleaner.run()


# ---------------------------------------------------------------------------
# Edge: extra unexpected columns are not a failure
# ---------------------------------------------------------------------------


def test_edge_extra_columns_not_a_failure(tmp_path: Path) -> None:
    """Edge: columns beyond the schema definition do not cause validation to fail."""
    # Arrange
    df = _valid_bronze_df().copy()
    df["totally_new_column"] = "unexpected"
    df["another_extra"] = 42
    cleaner = _make_cleaner(tmp_path, bronze_df=df)

    # Act / Assert — no exception
    cleaner.run()


# ---------------------------------------------------------------------------
# Edge: PULocationID = 1.0 (float repr of inclusive lower bound) is accepted
# ---------------------------------------------------------------------------


def test_edge_pulocationid_float_one_accepted(tmp_path: Path) -> None:
    """Edge: PULocationID = 1.0 is the inclusive lower bound and must be accepted."""
    # Arrange
    df = _valid_bronze_df().copy()
    df["PULocationID"] = 1.0  # all rows at the lower bound
    cleaner = _make_cleaner(tmp_path, bronze_df=df)

    # Act / Assert — no exception
    cleaner.run()


# ---------------------------------------------------------------------------
# Edge: TLC non-geographic zones 264 ("Unknown") and 265 ("NV") are accepted
# ---------------------------------------------------------------------------


def test_edge_pulocationid_264_accepted(tmp_path: Path) -> None:
    """Edge: PULocationID = 264 is a valid TLC raw code and must not be rejected.

    Zone 264 ('Unknown') appears in real NYC TLC data and is dropped
    downstream by DataCleaner (feature.py shapefile-orphan filter).
    The bronze schema must not block it before that stage runs.
    """
    # Arrange
    df = _valid_bronze_df().copy()
    df.loc[0, "PULocationID"] = 264.0
    cleaner = _make_cleaner(tmp_path, bronze_df=df)

    # Act / Assert — no exception
    cleaner.run()


def test_edge_pulocationid_265_accepted(tmp_path: Path) -> None:
    """Edge: PULocationID = 265 is a valid TLC raw code and must not be rejected.

    Zone 265 ('NV') appears in real NYC TLC data and is dropped downstream
    by DataCleaner. The bronze schema must not block it.
    """
    # Arrange
    df = _valid_bronze_df().copy()
    df.loc[0, "PULocationID"] = 265.0
    cleaner = _make_cleaner(tmp_path, bronze_df=df)

    # Act / Assert — no exception
    cleaner.run()


def test_edge_pulocationid_266_raises(tmp_path: Path) -> None:
    """Edge: PULocationID = 266 exceeds the highest known TLC zone and must raise."""
    # Arrange
    bad_df = _valid_bronze_df().copy()
    bad_df.loc[0, "PULocationID"] = 266.0
    cleaner = _make_cleaner(tmp_path, bronze_df=bad_df)

    # Act / Assert
    with pytest.raises(DataValidationError):
        cleaner.run()


# ---------------------------------------------------------------------------
# Edge: repeat runs produce distinct log files (no silent overwrite)
# ---------------------------------------------------------------------------


def test_edge_repeat_runs_produce_distinct_log_files(tmp_path: Path) -> None:
    """Edge: two successive run() calls write two distinct log files.

    The filename includes a seconds-precision timestamp.  To guarantee
    two distinct files we freeze time with monkeypatching at the schema
    module level so the filenames differ even in fast test environments.
    """
    import app.ingestion.schema as schema_module
    from datetime import datetime as _real_datetime
    from datetime import timezone as _tz

    timestamps = [
        _real_datetime(2024, 1, 1, 0, 0, 1, tzinfo=_tz.utc),
        _real_datetime(2024, 1, 1, 0, 0, 2, tzinfo=_tz.utc),
    ]
    call_index: list[int] = [0]

    class _FakeWallClock:
        """Fake WallClock that cycles through predetermined timestamps."""

        def now(self) -> _real_datetime:
            ts = timestamps[min(call_index[0], len(timestamps) - 1)]
            call_index[0] += 1
            return ts

    original_wallclock = schema_module.WallClock
    schema_module.WallClock = _FakeWallClock  # type: ignore[misc,assignment]

    try:
        bronze_df = _valid_bronze_df()
        # Both runs share the same _validation dir for the count check — use a
        # common silver parent dir rather than the per-cleaner default layout.
        shared_silver = tmp_path / "shared_silver"
        shared_silver.mkdir()

        def _shared_config(run_label: str) -> dict[str, Any]:
            bronze_p = tmp_path / run_label / "raw.parquet"
            bronze_p.parent.mkdir(parents=True, exist_ok=True)
            bronze_df.to_parquet(bronze_p, index=False)
            return {
                "env": "local",
                "data": {
                    "bronze_path": str(bronze_p),
                    "silver_path": str(shared_silver / "cleaned.parquet"),
                },
                "clean": {"null_strategy": "drop"},
                "inference": {"simulation": {"start_hour": "2023-06-01 00:00:00"}},
                "model": {"store_path": str(tmp_path / "models")},
            }

        c1 = DataCleaner(_shared_config("run1"))
        call_index[0] = 0
        c1.run()

        call_index[0] = 1
        c2 = DataCleaner(_shared_config("run2"))
        c2.run()

    finally:
        schema_module.WallClock = original_wallclock  # type: ignore[misc,assignment]

    # Assert — two distinct log files
    log_dir = shared_silver / "_validation"
    log_files = sorted(log_dir.glob("validation_*.json"))
    assert len(log_files) == 2, (
        f"Expected 2 distinct log files after two runs, got {len(log_files)}"
    )
    assert log_files[0].name != log_files[1].name, (
        "The two log file names must differ (no silent overwrite)"
    )


# ---------------------------------------------------------------------------
# Edge: empty Bronze frame passes schema (no rows = no violations)
# ---------------------------------------------------------------------------


def test_edge_empty_bronze_frame_passes_schema(tmp_path: Path) -> None:
    """Edge: an empty Bronze frame (zero rows, correct columns) passes validation."""
    # Arrange — same columns but zero rows
    empty_df = _valid_bronze_df().iloc[0:0].copy()
    cleaner = _make_cleaner(tmp_path, bronze_df=empty_df)

    # Act / Assert — no exception
    cleaner.run()

    # And the log was written
    log_dir = _validation_dir(tmp_path)
    payload = _read_single_log(log_dir)
    assert payload["rows_in"] == 0
    assert payload["rows_passing"] == 0
    assert payload["rows_failing"] == 0


# ---------------------------------------------------------------------------
# Step 3 — zone_id_min/max via build_bronze_schema and validate_bronze_schema
# ---------------------------------------------------------------------------


def test_build_bronze_schema_accepts_locationid_within_custom_range() -> None:
    """build_bronze_schema(1, 300) accepts PULocationID=280 that 265 would reject."""
    # Arrange
    df = _valid_bronze_df().copy()
    df.loc[0, "PULocationID"] = 280.0
    schema = build_bronze_schema(1, 300)

    # Act / Assert — no SchemaErrors raised
    schema.validate(df, lazy=True)


def test_validate_bronze_schema_config_override_widens_zone_range() -> None:
    """validate_bronze_schema(df, config=...) with zone_id_max=300 accepts ID=280."""
    # Arrange
    df = _valid_bronze_df().copy()
    df.loc[0, "PULocationID"] = 280.0
    config = {"ingest": {"zone_id_min": 1, "zone_id_max": 300}}

    # Act / Assert — no exception with widened range
    validate_bronze_schema(df, config=config)


def test_validate_bronze_schema_default_still_rejects_zone_266() -> None:
    """Default zone_id_max=265 still rejects PULocationID=266 (regression guard)."""
    # Arrange
    df = _valid_bronze_df().copy()
    df.loc[0, "PULocationID"] = 266.0
    # No config override — should use the 265 default

    # Act / Assert
    with pytest.raises(DataValidationError):
        validate_bronze_schema(df)
