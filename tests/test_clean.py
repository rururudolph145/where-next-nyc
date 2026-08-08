import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

import app.ingestion.schema as schema_module
from app.ingestion.clean import DataCleaner
from app.ingestion.schema import DataValidationError, validate_bronze_schema


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": [
                "2023-01-01 08:00:00",
                "2023-01-01 09:00:00",
                None,
            ],
            "tpep_dropoff_datetime": [
                "2023-01-01 08:15:00",
                "2023-01-01 09:20:00",
                None,
            ],
            "passenger_count": [2, 1, 0],
            "trip_distance": [1.5, 3.2, 0.0],
            "fare_amount": [7.5, 12.0, 5.0],
            "PULocationID": [1, 2, 3],
        }
    )


@pytest.fixture
def cleaner():
    config = {
        "env": "local",
        "data": {
            "bronze_path": "data/bronze/test.parquet",
            "silver_path": "data/silver/test.parquet",
        },
        "clean": {"null_strategy": "drop"},
        "inference": {"simulation": {"start_hour": "2023-06-01 00:00:00"}},
        "model": {"store_path": "models/"},
    }
    return DataCleaner(config)


def test_clean_data(sample_data, cleaner):
    result, bounds = cleaner.clean(sample_data)

    # Rows with null datetimes or non-positive passenger/distance should be dropped
    assert result["tpep_pickup_datetime"].notna().all()
    assert (result["passenger_count"] > 0).all()
    assert (result["trip_distance"] > 0).all()

    # Time features should be created
    assert "hour" in result.columns
    assert "day_of_week" in result.columns
    assert "month" in result.columns
    assert "is_weekend" in result.columns

    # Bounds should be returned
    assert "q1" in bounds
    assert "q3" in bounds
    assert "iqr" in bounds
    assert bounds["iqr"] == bounds["q3"] - bounds["q1"]


def test_clean_transform_mode(cleaner):
    """Verify that provided IQR bounds are used instead of recomputed."""
    data = pd.DataFrame(
        {
            "tpep_pickup_datetime": [
                "2023-01-01 08:00:00",
                "2023-01-01 09:00:00",
                "2023-01-01 10:00:00",
            ],
            "tpep_dropoff_datetime": [
                "2023-01-01 08:15:00",
                "2023-01-01 09:20:00",
                "2023-01-01 10:30:00",
            ],
            "passenger_count": [1, 1, 1],
            "trip_distance": [2.0, 3.0, 4.0],
            "fare_amount": [10.0, 20.0, 100.0],
            "PULocationID": [1, 2, 3],
        }
    )

    # Fit mode — bounds computed from data
    _, fit_bounds = cleaner.clean(data)

    # Transform mode — use tight bounds that exclude the 100.0 fare
    tight_bounds = {"q1": 10.0, "q3": 20.0, "iqr": 10.0}
    result, returned_bounds = cleaner.clean(data, iqr_bounds=tight_bounds)

    # The returned bounds should be exactly what we passed in
    assert returned_bounds == tight_bounds

    # The 100.0 fare row should be filtered out with tight bounds
    assert (result["fare_amount"] <= 20.0 + 1.5 * 10.0).all()
    assert 100.0 not in result["fare_amount"].values


def test_clean_iqr_multiplier_from_config_survives_wider_fence() -> None:
    """iqr_multiplier=3.0 keeps a row that default 1.5 would drop.

    Data: fares [10, 12, 14, 16, 25].
    q1=12.0, q3=16.0, iqr=4.0 → 1.5 fence upper=22.0; 3.0 fence upper=28.0.
    fare=25.0 is outside the 1.5 fence but inside the 3.0 fence.
    """
    data = pd.DataFrame(
        {
            "tpep_pickup_datetime": ["2023-01-01 08:00:00"] * 5,
            "tpep_dropoff_datetime": ["2023-01-01 08:15:00"] * 5,
            "passenger_count": [1.0] * 5,
            "trip_distance": [1.0] * 5,
            "fare_amount": [10.0, 12.0, 14.0, 16.0, 25.0],
            "PULocationID": [1, 2, 3, 4, 5],
        }
    )
    config = {
        "env": "local",
        "data": {"bronze_path": "x", "silver_path": "y"},
        "clean": {"null_strategy": "drop", "iqr_multiplier": 3.0},
        "inference": {"simulation": {"start_hour": "2023-06-01 00:00:00"}},
        "model": {"store_path": "models/"},
    }
    cleaner = DataCleaner(config)

    # Act
    result, _ = cleaner.clean(data)

    # Assert — row with fare 25.0 survives the wider 3.0 fence
    assert 25.0 in result["fare_amount"].values


def test_clean_iqr_multiplier_default_drops_outlier() -> None:
    """Default iqr_multiplier=1.5 (no config key) drops a row that 3.0 would keep.

    Same data as above; fare=25.0 is outside the 1.5 fence upper (22.0).
    """
    data = pd.DataFrame(
        {
            "tpep_pickup_datetime": ["2023-01-01 08:00:00"] * 5,
            "tpep_dropoff_datetime": ["2023-01-01 08:15:00"] * 5,
            "passenger_count": [1.0] * 5,
            "trip_distance": [1.0] * 5,
            "fare_amount": [10.0, 12.0, 14.0, 16.0, 25.0],
            "PULocationID": [1, 2, 3, 4, 5],
        }
    )
    config = {
        "env": "local",
        "data": {"bronze_path": "x", "silver_path": "y"},
        "clean": {"null_strategy": "drop"},  # no iqr_multiplier key — uses default 1.5
        "inference": {"simulation": {"start_hour": "2023-06-01 00:00:00"}},
        "model": {"store_path": "models/"},
    }
    cleaner = DataCleaner(config)

    # Act
    result, _ = cleaner.clean(data)

    # Assert — row with fare 25.0 is dropped by the default 1.5 fence (upper=22.0)
    assert 25.0 not in result["fare_amount"].values


# ---------------------------------------------------------------------------
# Fixtures for schema-validation tests
# ---------------------------------------------------------------------------


def _valid_bronze_df() -> pd.DataFrame:
    """Minimal valid Bronze frame with all required columns and types."""
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.array(
                ["2023-01-01 08:00:00", "2023-01-01 09:00:00"],
                dtype="datetime64[us]",
            ),
            "tpep_dropoff_datetime": pd.array(
                ["2023-01-01 08:15:00", "2023-01-01 09:20:00"],
                dtype="datetime64[us]",
            ),
            "PULocationID": pd.array([1.0, 100.0], dtype="float64"),
            "passenger_count": pd.array([2.0, 1.0], dtype="float64"),
            "trip_distance": pd.array([1.5, 3.2], dtype="float64"),
            "fare_amount": pd.array([7.5, 12.0], dtype="float64"),
        }
    )


@pytest.fixture
def tmp_validation_config(tmp_path: Path) -> dict[str, Any]:
    """Config dict whose silver_path lives inside tmp_path."""
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    return {
        "env": "local",
        "data": {
            "bronze_path": str(tmp_path / "bronze.parquet"),
            "silver_path": str(silver_dir / "cleaned.parquet"),
        },
        "clean": {"null_strategy": "drop"},
        "inference": {"simulation": {"start_hour": "2023-06-01 00:00:00"}},
        "model": {"store_path": str(tmp_path / "models")},
    }


# ---------------------------------------------------------------------------
# TestSchemaValidation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """AC coverage for PBI 19 — Bronze schema validation checkpoint."""

    # ------------------------------------------------------------------
    # validate_bronze_schema — unit tests (pure: no filesystem I/O)
    # ------------------------------------------------------------------

    def test_validate_bronze_schema_valid_frame_no_exception(self) -> None:
        """AC #1: valid frame passes silently."""
        # Arrange
        df = _valid_bronze_df()
        # Act / Assert — no exception raised
        validate_bronze_schema(df)

    def test_validate_bronze_schema_missing_column_raises(self) -> None:
        """AC #2 / #8b: missing required column raises DataValidationError."""
        # Arrange
        df = _valid_bronze_df().drop(columns=["PULocationID"])
        # Act / Assert
        with pytest.raises(DataValidationError):
            validate_bronze_schema(df)

    def test_validate_bronze_schema_pulocationid_out_of_range_raises(
        self,
    ) -> None:
        """AC #3 / #8c: PULocationID = 266.0 (above highest TLC zone) raises DataValidationError."""
        # Arrange
        df = _valid_bronze_df().copy()
        df.loc[0, "PULocationID"] = 266.0
        # Act / Assert
        with pytest.raises(DataValidationError):
            validate_bronze_schema(df)

    def test_validate_bronze_schema_extra_columns_no_exception(self) -> None:
        """Edge: additional columns beyond the schema are allowed."""
        # Arrange
        df = _valid_bronze_df().copy()
        df["extra_col"] = "hello"
        # Act / Assert — no exception
        validate_bronze_schema(df)

    def test_validate_bronze_schema_pulocationid_float_one_passes(
        self,
    ) -> None:
        """Edge: PULocationID = 1.0 is the inclusive lower bound — must pass."""
        # Arrange
        df = _valid_bronze_df().copy()
        df["PULocationID"] = 1.0
        # Act / Assert — no exception
        validate_bronze_schema(df)

    def test_validate_bronze_schema_datetime_ns_resolution_passes(
        self,
    ) -> None:
        """Edge: datetime64[ns] column resolution is accepted."""
        # Arrange
        df = _valid_bronze_df().copy()
        df["tpep_pickup_datetime"] = df["tpep_pickup_datetime"].astype("datetime64[ns]")
        # Act / Assert — no exception
        validate_bronze_schema(df)

    # ------------------------------------------------------------------
    # DataValidationError type check
    # ------------------------------------------------------------------

    def test_datavalidation_error_is_not_value_error(self) -> None:
        """AC #6: DataValidationError must NOT be a subclass of ValueError."""
        assert not issubclass(DataValidationError, ValueError)

    # ------------------------------------------------------------------
    # DataCleaner.run() integration tests (mocked I/O)
    # ------------------------------------------------------------------

    def test_run_valid_frame_log_written_with_required_fields(
        self, tmp_validation_config: dict[str, Any]
    ) -> None:
        """AC #4 / #5 / #8d: clean pass writes log with all five required keys."""
        # Arrange
        cleaner = DataCleaner(tmp_validation_config)
        df = _valid_bronze_df()

        with (
            patch.object(cleaner, "load_bronze", return_value=df),
            patch.object(cleaner, "save_silver"),
            patch(
                "app.ingestion.clean.write_validation_log",
                wraps=schema_module.write_validation_log,
            ),
        ):
            # Act
            cleaner.run()

        # Assert — locate the log file
        log_dir = (
            Path(tmp_validation_config["data"]["silver_path"]).parent / "_validation"
        )
        log_files = list(log_dir.glob("validation_*.json"))
        assert len(log_files) == 1
        with open(log_files[0]) as fh:
            payload = json.load(fh)
        required_keys = {
            "validated_at",
            "rows_in",
            "rows_passing",
            "rows_failing",
            "failure_reasons",
        }
        assert required_keys.issubset(payload.keys())

    def test_run_log_written_even_on_validation_failure(
        self, tmp_validation_config: dict[str, Any]
    ) -> None:
        """AC #4: log file is written even when validation fails."""
        # Arrange — frame missing PULocationID
        cleaner = DataCleaner(tmp_validation_config)
        df = _valid_bronze_df().drop(columns=["PULocationID"])

        with (
            patch.object(cleaner, "load_bronze", return_value=df),
            patch.object(cleaner, "save_silver"),
        ):
            # Act
            with pytest.raises(DataValidationError):
                cleaner.run()

        # Assert — log still written
        log_dir = (
            Path(tmp_validation_config["data"]["silver_path"]).parent / "_validation"
        )
        log_files = list(log_dir.glob("validation_*.json"))
        assert len(log_files) == 1

    def test_run_repeat_calls_produce_separate_log_files(
        self,
        tmp_validation_config: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Edge: two successive calls write two distinct log files (no overwrite)."""
        # Arrange — use a fake WallClock that returns two distinct timestamps
        t1 = datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
        t2 = datetime(2024, 1, 1, 0, 0, 2, tzinfo=timezone.utc)
        call_count = {"n": 0}

        class _FakeWallClock:
            def now(self) -> datetime:
                call_count["n"] += 1
                return t1 if call_count["n"] <= 2 else t2

        # Patch WallClock inside schema module so the default clock
        # created by write_validation_log() returns controlled timestamps.
        monkeypatch.setattr(schema_module, "WallClock", _FakeWallClock)

        cleaner = DataCleaner(tmp_validation_config)
        df = _valid_bronze_df()

        with (
            patch.object(cleaner, "load_bronze", return_value=df),
            patch.object(cleaner, "save_silver"),
        ):
            # Act — first call
            cleaner.run()

        # Reset call count and patch new timestamp for second call
        call_count["n"] = 2  # ensure next call returns t2

        with (
            patch.object(cleaner, "load_bronze", return_value=df),
            patch.object(cleaner, "save_silver"),
        ):
            cleaner.run()

        # Assert
        log_dir = (
            Path(tmp_validation_config["data"]["silver_path"]).parent / "_validation"
        )
        log_files = sorted(log_dir.glob("validation_*.json"))
        assert len(log_files) == 2
