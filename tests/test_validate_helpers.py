"""Tests for orchestration/lib/validate_helpers.py.

PBI EXP-03a (experimentation) — the pure helpers the validate DAG delegates to.
Inserts a ``validate`` task between ``ingest`` and ``advance_watermark`` so
malformed bronze raises ``RuntimeError`` → ``advance``/``write`` skip under
``all_success`` → watermark held. Every run (pass OR fail) writes a
``validation_<timestamp>.json`` log under the experiment silver root.

The logic lives in a PURE, Airflow-free module (the experiment container has
Airflow; the host test env does NOT). Two seams keep it unit-testable:

  * ``build_validate_config`` — deep-overrides ``data.silver_path`` so
    ``validation_log_dir(config)`` resolves under the experiment root (never
    production ``data/``). Nothing else changes.
  * ``run_validate`` — reads bronze parquet, calls ``validate_bronze_schema``,
    writes a log via ``write_validation_log`` inside ``finally`` (so failures
    are logged too), re-raises ``DataValidationError`` as ``RuntimeError`` for
    Airflow.

AC coverage (from PLAN_EXP03a §Implementation Steps 1-3):
  - build_validate_config overrides data.silver_path
  - build_validate_config does not mutate the base config (deep copy)
  - build_validate_config preserves unrelated keys
  - run_validate reads bronze from the given path
  - run_validate calls validate_bronze_schema with the loaded DataFrame
  - run_validate writes a log with empty failure_reasons on the happy path
  - run_validate returns the bronze path on success (stable XCom flow)
  - run_validate re-raises DataValidationError as RuntimeError
  - run_validate writes a log with failure_reasons BEFORE re-raising
  - run_validate threads the injected Clock through to write_validation_log
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from app.core.clock import SimulatedClock
from app.ingestion.schema import DataValidationError
from orchestration.lib.validate_helpers import (
    build_validate_config,
    run_validate,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

BRONZE_PATH = "/opt/airflow/data_exp/bronze/yellow_tripdata/bronze.parquet"
SILVER_ROOT = "/opt/airflow/data_exp/silver/yellow_tripdata_cleaned.parquet"


@pytest.fixture
def base_config() -> dict[str, Any]:
    """Minimal config mirroring the shape ``validation_log_dir`` reads."""
    return {
        "ingest": {
            "slice_duration": "15min",
            "bronze_schema": {"tpep_pickup_datetime": "datetime64[us]"},
        },
        "feature": {"bucket": "15min"},
        "data": {
            "raw_path": "data/raw/yellow_tripdata",
            "bronze_path": "data/bronze/yellow_tripdata/bronze.parquet",
            "silver_path": "data/silver/yellow_tripdata_cleaned.parquet",
        },
    }


@pytest.fixture
def override_config(base_config: dict[str, Any]) -> dict[str, Any]:
    """A config already pointed at the experiment silver root."""
    return build_validate_config(base_config, silver_root=SILVER_ROOT)


# ---------------------------------------------------------------------------
# TestBuildValidateConfig
# ---------------------------------------------------------------------------


class TestBuildValidateConfig:
    """build_validate_config injects the isolated silver path (the seam)."""

    def test_overrides_silver_path(self, base_config: dict[str, Any]) -> None:
        """data.silver_path points at the injected experiment silver root."""
        # Act
        result = build_validate_config(base_config, silver_root=SILVER_ROOT)

        # Assert
        assert result["data"]["silver_path"] == SILVER_ROOT

    def test_does_not_mutate_base(self, base_config: dict[str, Any]) -> None:
        """The base config is deep-copied, never mutated in place."""
        # Arrange
        snapshot = copy.deepcopy(base_config)

        # Act
        build_validate_config(base_config, silver_root=SILVER_ROOT)

        # Assert
        assert base_config == snapshot

    def test_preserves_unrelated_keys(self, base_config: dict[str, Any]) -> None:
        """Only data.silver_path changes; everything else passes through."""
        # Act
        result = build_validate_config(base_config, silver_root=SILVER_ROOT)

        # Assert
        assert result["data"]["raw_path"] == base_config["data"]["raw_path"]
        assert result["data"]["bronze_path"] == base_config["data"]["bronze_path"]
        assert result["ingest"] == base_config["ingest"]
        assert result["feature"] == base_config["feature"]


# ---------------------------------------------------------------------------
# TestRunValidateHappyPath
# ---------------------------------------------------------------------------


class TestRunValidateHappyPath:
    """run_validate loads bronze, validates it, and logs a clean pass."""

    def test_reads_bronze_from_given_path(
        self, override_config: dict[str, Any]
    ) -> None:
        """pd.read_parquet is called with the bronze_path arg unchanged."""
        # Arrange
        df = pd.DataFrame({"PULocationID": [1.0]})
        with (
            patch(
                "orchestration.lib.validate_helpers.pd.read_parquet",
                return_value=df,
            ) as mock_read,
            patch("orchestration.lib.validate_helpers.validate_bronze_schema"),
            patch("orchestration.lib.validate_helpers.write_validation_log"),
        ):
            # Act
            run_validate(BRONZE_PATH, config=override_config)

        # Assert
        mock_read.assert_called_once_with(BRONZE_PATH)

    def test_calls_validate_bronze_schema_with_dataframe(
        self, override_config: dict[str, Any]
    ) -> None:
        """The DataFrame returned by read_parquet is what gets validated."""
        # Arrange
        df = pd.DataFrame({"PULocationID": [1.0]})
        with (
            patch(
                "orchestration.lib.validate_helpers.pd.read_parquet",
                return_value=df,
            ),
            patch(
                "orchestration.lib.validate_helpers.validate_bronze_schema"
            ) as mock_validate,
            patch("orchestration.lib.validate_helpers.write_validation_log"),
        ):
            # Act
            run_validate(BRONZE_PATH, config=override_config)

        # Assert
        mock_validate.assert_called_once()
        (received_df,), _ = mock_validate.call_args
        assert received_df is df

    def test_writes_log_with_empty_failure_reasons(
        self, override_config: dict[str, Any]
    ) -> None:
        """Happy-path log carries failure_reasons=[] (rows_failing==0)."""
        # Arrange
        df = pd.DataFrame({"PULocationID": [1.0]})
        with (
            patch(
                "orchestration.lib.validate_helpers.pd.read_parquet",
                return_value=df,
            ),
            patch("orchestration.lib.validate_helpers.validate_bronze_schema"),
            patch(
                "orchestration.lib.validate_helpers.write_validation_log"
            ) as mock_log,
        ):
            # Act
            run_validate(BRONZE_PATH, config=override_config)

        # Assert
        mock_log.assert_called_once()
        args, _ = mock_log.call_args
        # Positional args: (log_dir, df, failure_reasons)
        assert args[2] == []
        # log_dir resolves under the overridden silver root
        assert str(args[0]).endswith("/silver/_validation")

    def test_returns_bronze_path_on_success(
        self, override_config: dict[str, Any]
    ) -> None:
        """A stable XCom contract: run_validate hands the bronze path forward."""
        # Arrange
        df = pd.DataFrame({"PULocationID": [1.0]})
        with (
            patch(
                "orchestration.lib.validate_helpers.pd.read_parquet",
                return_value=df,
            ),
            patch("orchestration.lib.validate_helpers.validate_bronze_schema"),
            patch("orchestration.lib.validate_helpers.write_validation_log"),
        ):
            # Act
            result = run_validate(BRONZE_PATH, config=override_config)

        # Assert
        assert result == BRONZE_PATH


# ---------------------------------------------------------------------------
# TestRunValidateFailurePath
# ---------------------------------------------------------------------------


class TestRunValidateFailurePath:
    """Malformed bronze → RuntimeError, but the log is written first."""

    def test_reraises_data_validation_error_as_runtime_error(
        self, override_config: dict[str, Any]
    ) -> None:
        """DataValidationError becomes RuntimeError so Airflow marks it failed."""
        # Arrange
        df = pd.DataFrame({"PULocationID": [999.0]})
        reasons = ["PULocationID out of range [1, 263]"]
        with (
            patch(
                "orchestration.lib.validate_helpers.pd.read_parquet",
                return_value=df,
            ),
            patch(
                "orchestration.lib.validate_helpers.validate_bronze_schema",
                side_effect=DataValidationError(reasons),
            ),
            patch("orchestration.lib.validate_helpers.write_validation_log"),
        ):
            # Act / Assert
            with pytest.raises(RuntimeError) as exc_info:
                run_validate(BRONZE_PATH, config=override_config)

        assert "PULocationID out of range" in str(exc_info.value)

    def test_writes_log_with_failure_reasons_before_reraising(
        self, override_config: dict[str, Any]
    ) -> None:
        """The finally-block log write fires even when validate raises."""
        # Arrange
        df = pd.DataFrame({"PULocationID": [999.0]})
        reasons = ["PULocationID out of range [1, 263]"]
        with (
            patch(
                "orchestration.lib.validate_helpers.pd.read_parquet",
                return_value=df,
            ),
            patch(
                "orchestration.lib.validate_helpers.validate_bronze_schema",
                side_effect=DataValidationError(reasons),
            ),
            patch(
                "orchestration.lib.validate_helpers.write_validation_log"
            ) as mock_log,
        ):
            # Act
            with pytest.raises(RuntimeError):
                run_validate(BRONZE_PATH, config=override_config)

        # Assert — log written exactly once with the failure reasons
        mock_log.assert_called_once()
        args, _ = mock_log.call_args
        assert args[2] == reasons  # failure_reasons positional arg

    def test_clock_passed_through_to_log_writer(
        self, override_config: dict[str, Any]
    ) -> None:
        """A caller-supplied Clock reaches write_validation_log unchanged."""
        # Arrange
        df = pd.DataFrame({"PULocationID": [1.0]})
        clock = SimulatedClock(datetime(2021, 1, 18, 0, 15, 0, tzinfo=timezone.utc))
        with (
            patch(
                "orchestration.lib.validate_helpers.pd.read_parquet",
                return_value=df,
            ),
            patch("orchestration.lib.validate_helpers.validate_bronze_schema"),
            patch(
                "orchestration.lib.validate_helpers.write_validation_log"
            ) as mock_log,
        ):
            # Act
            run_validate(BRONZE_PATH, config=override_config, clock=clock)

        # Assert
        _, kwargs = mock_log.call_args
        assert kwargs.get("clock") is clock
