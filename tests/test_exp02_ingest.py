"""Tests for orchestration_experiment/dags/exp02_ingest.py.

PBI EXP-02 (experimentation) — the pure helpers the ingest DAG delegates to.
Inserts an ``ingest`` task between ``read_watermark`` and ``advance_watermark``
so a run writes CUMULATIVE bronze ``[seed_start, window_end)`` via the real
``app.ingestion.Ingestion``.

The logic lives in a PURE, Airflow-free module (the experiment container has
Airflow; the host test env does NOT). Two seams keep it unit-testable:

  * ``build_ingest_config`` — deep-overrides ``data.raw_path`` /
    ``data.bronze_path`` so ``Ingestion`` reads raw from a read-only root and
    writes bronze under the isolated experiment root (never production
    ``data/``). It must preserve ``ingest.slice_duration == feature.bucket`` or
    ``Ingestion.__init__`` asserts.
  * ``run_ingest`` — constructs ``Ingestion(config=...)`` and calls
    ``run(window_start=seed_start, window_end=window_end)`` with the seed FROZEN
    as the lower bound (cumulative bronze). Verified against a MOCKED
    ``Ingestion`` — the 340MB raw read happens only in the container smoke
    test, never in pytest (testing rule: no external data).

AC coverage (from PLAN_EXP02 §Implementation Steps 1-2):
  - build_ingest_config points raw_path/bronze_path at the injected roots
  - build_ingest_config preserves slice_duration == bucket (init guard)
  - build_ingest_config does not mutate the base config (deep copy)
  - run_ingest calls run() with the FROZEN seed as window_start
  - run_ingest passes the advancing window_end through unchanged
  - run_ingest returns the bronze path Ingestion.run() reports
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any
from unittest.mock import patch

import pytest

from orchestration_experiment.dags.exp02_ingest import (
    build_ingest_config,
    run_ingest,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

SEED_START = datetime(2021, 1, 18, 0, 0, 0)  # naive NYC wall-clock; from config
RAW_ROOT = "/opt/airflow/data/raw/yellow_tripdata"
BRONZE_ROOT = "/opt/airflow/data_exp/bronze/yellow_tripdata/bronze.parquet"


@pytest.fixture
def base_config() -> dict[str, Any]:
    """A minimal config mirroring the shape Ingestion.__init__ reads."""
    return {
        "ingest": {
            "start_year_month": "2021-01",
            "end_year_month": "2021-03",
            "slice_duration": "15min",
            "bronze_schema": {"tpep_pickup_datetime": "datetime64[us]"},
        },
        "feature": {"bucket": "15min"},
        "data": {
            "raw_path": "data/raw/yellow_tripdata",
            "bronze_path": "data/bronze/yellow_tripdata/bronze.parquet",
        },
    }


# ---------------------------------------------------------------------------
# TestBuildIngestConfig
# ---------------------------------------------------------------------------


class TestBuildIngestConfig:
    """build_ingest_config injects isolated experiment paths (the seam)."""

    def test_overrides_raw_path(self, base_config: dict[str, Any]) -> None:
        """data.raw_path points at the injected read-only raw root."""
        # Act
        result = build_ingest_config(
            base_config, raw_root=RAW_ROOT, bronze_root=BRONZE_ROOT
        )

        # Assert
        assert result["data"]["raw_path"] == RAW_ROOT

    def test_overrides_bronze_path(self, base_config: dict[str, Any]) -> None:
        """data.bronze_path points at the isolated experiment bronze root."""
        # Act
        result = build_ingest_config(
            base_config, raw_root=RAW_ROOT, bronze_root=BRONZE_ROOT
        )

        # Assert
        assert result["data"]["bronze_path"] == BRONZE_ROOT

    def test_preserves_slice_equals_bucket(self, base_config: dict[str, Any]) -> None:
        """The init guard (slice_duration == bucket) must survive the override."""
        # Act
        result = build_ingest_config(
            base_config, raw_root=RAW_ROOT, bronze_root=BRONZE_ROOT
        )

        # Assert
        assert result["ingest"]["slice_duration"] == result["feature"]["bucket"]

    def test_does_not_mutate_base(self, base_config: dict[str, Any]) -> None:
        """The base config is deep-copied, never mutated in place."""
        # Arrange
        snapshot = copy.deepcopy(base_config)

        # Act
        build_ingest_config(base_config, raw_root=RAW_ROOT, bronze_root=BRONZE_ROOT)

        # Assert
        assert base_config == snapshot

    def test_preserves_unrelated_keys(self, base_config: dict[str, Any]) -> None:
        """Keys outside data.* pass through untouched (only paths change)."""
        # Act
        result = build_ingest_config(
            base_config, raw_root=RAW_ROOT, bronze_root=BRONZE_ROOT
        )

        # Assert
        assert result["ingest"]["start_year_month"] == "2021-01"
        assert (
            result["ingest"]["bronze_schema"] == base_config["ingest"]["bronze_schema"]
        )


# ---------------------------------------------------------------------------
# TestRunIngest
# ---------------------------------------------------------------------------


class TestRunIngest:
    """run_ingest calls Ingestion with the FROZEN seed as the lower bound."""

    def test_freezes_seed_as_window_start(self, base_config: dict[str, Any]) -> None:
        """Cumulative bronze: window_start is ALWAYS the seed, never window_end."""
        # Arrange
        window_end = datetime(2021, 1, 18, 0, 15, 0)
        with patch("orchestration_experiment.dags.exp02_ingest.Ingestion") as mock_cls:
            instance = mock_cls.return_value
            instance.run.return_value = BRONZE_ROOT

            # Act
            run_ingest(window_end, seed_start=SEED_START, config=base_config)

        # Assert
        _, kwargs = instance.run.call_args
        assert kwargs["window_start"] == SEED_START

    def test_passes_window_end_through(self, base_config: dict[str, Any]) -> None:
        """The advancing upper bound is passed to run() unchanged."""
        # Arrange
        window_end = datetime(2021, 1, 18, 0, 30, 0)
        with patch("orchestration_experiment.dags.exp02_ingest.Ingestion") as mock_cls:
            instance = mock_cls.return_value
            instance.run.return_value = BRONZE_ROOT

            # Act
            run_ingest(window_end, seed_start=SEED_START, config=base_config)

        # Assert
        _, kwargs = instance.run.call_args
        assert kwargs["window_end"] == window_end

    def test_constructs_ingestion_with_injected_config(
        self, base_config: dict[str, Any]
    ) -> None:
        """Ingestion is built from the injected config (path isolation seam)."""
        # Arrange
        window_end = datetime(2021, 1, 18, 0, 15, 0)
        with patch("orchestration_experiment.dags.exp02_ingest.Ingestion") as mock_cls:
            mock_cls.return_value.run.return_value = BRONZE_ROOT

            # Act
            run_ingest(window_end, seed_start=SEED_START, config=base_config)

        # Assert
        _, kwargs = mock_cls.call_args
        assert kwargs["config"] is base_config

    def test_returns_bronze_path(self, base_config: dict[str, Any]) -> None:
        """run_ingest returns the path Ingestion.run() reports."""
        # Arrange
        window_end = datetime(2021, 1, 18, 0, 15, 0)
        with patch("orchestration_experiment.dags.exp02_ingest.Ingestion") as mock_cls:
            mock_cls.return_value.run.return_value = BRONZE_ROOT

            # Act
            result = run_ingest(window_end, seed_start=SEED_START, config=base_config)

        # Assert
        assert result == BRONZE_ROOT

    def test_run_called_exactly_once(self, base_config: dict[str, Any]) -> None:
        """A single run() call per invocation — no accidental double-ingest."""
        # Arrange
        window_end = datetime(2021, 1, 18, 0, 15, 0)
        with patch("orchestration_experiment.dags.exp02_ingest.Ingestion") as mock_cls:
            instance = mock_cls.return_value
            instance.run.return_value = BRONZE_ROOT

            # Act
            run_ingest(window_end, seed_start=SEED_START, config=base_config)

        # Assert
        assert instance.run.call_count == 1
