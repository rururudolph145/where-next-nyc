"""Tests for orchestration/lib/ingest_helpers.py.

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

from datetime import datetime
from unittest.mock import patch

import pytest

from app.core.config import AppConfig
from orchestration.lib.ingest_helpers import (
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
def base_config() -> AppConfig:
    """AppConfig mirroring the shape Ingestion.__init__ reads."""
    from tests.conftest import _FULL_CONFIG_DICT

    return AppConfig.model_validate(_FULL_CONFIG_DICT)


# ---------------------------------------------------------------------------
# TestBuildIngestConfig
# ---------------------------------------------------------------------------


class TestBuildIngestConfig:
    """build_ingest_config injects isolated experiment paths (the seam)."""

    def test_overrides_raw_path(self, base_config: AppConfig) -> None:
        """data.raw_path points at the injected read-only raw root."""
        # Act
        result = build_ingest_config(
            base_config, raw_root=RAW_ROOT, bronze_root=BRONZE_ROOT
        )

        # Assert
        assert result.data.raw_path == RAW_ROOT

    def test_overrides_bronze_path(self, base_config: AppConfig) -> None:
        """data.bronze_path points at the isolated experiment bronze root."""
        # Act
        result = build_ingest_config(
            base_config, raw_root=RAW_ROOT, bronze_root=BRONZE_ROOT
        )

        # Assert
        assert result.data.bronze_path == BRONZE_ROOT

    def test_preserves_slice_equals_bucket(self, base_config: AppConfig) -> None:
        """The init guard (slice_duration == bucket) must survive the override."""
        # Act
        result = build_ingest_config(
            base_config, raw_root=RAW_ROOT, bronze_root=BRONZE_ROOT
        )

        # Assert
        assert result.ingest.slice_duration == result.feature.bucket

    def test_does_not_mutate_base(self, base_config: AppConfig) -> None:
        """Frozen model: the returned config is a new object, not base_config."""
        # Act
        result = build_ingest_config(
            base_config, raw_root=RAW_ROOT, bronze_root=BRONZE_ROOT
        )

        # Assert
        assert result is not base_config

    def test_preserves_unrelated_keys(self, base_config: AppConfig) -> None:
        """Keys outside data.* pass through untouched (only paths change)."""
        # Act
        result = build_ingest_config(
            base_config, raw_root=RAW_ROOT, bronze_root=BRONZE_ROOT
        )

        # Assert
        assert result.ingest.start_year_month == "2021-01"
        assert result.ingest.bronze_schema == base_config.ingest.bronze_schema


# ---------------------------------------------------------------------------
# TestRunIngest
# ---------------------------------------------------------------------------


class TestRunIngest:
    """run_ingest calls Ingestion with the FROZEN seed as the lower bound."""

    def test_freezes_seed_as_window_start(self, base_config: AppConfig) -> None:
        """Cumulative bronze: window_start is ALWAYS the seed, never window_end."""
        # Arrange
        window_end = datetime(2021, 1, 18, 0, 15, 0)
        with patch("orchestration.lib.ingest_helpers.Ingestion") as mock_cls:
            instance = mock_cls.return_value
            instance.run.return_value = BRONZE_ROOT

            # Act
            run_ingest(window_end, seed_start=SEED_START, config=base_config)

        # Assert
        _, kwargs = instance.run.call_args
        assert kwargs["window_start"] == SEED_START

    def test_passes_window_end_through(self, base_config: AppConfig) -> None:
        """The advancing upper bound is passed to run() unchanged."""
        # Arrange
        window_end = datetime(2021, 1, 18, 0, 30, 0)
        with patch("orchestration.lib.ingest_helpers.Ingestion") as mock_cls:
            instance = mock_cls.return_value
            instance.run.return_value = BRONZE_ROOT

            # Act
            run_ingest(window_end, seed_start=SEED_START, config=base_config)

        # Assert
        _, kwargs = instance.run.call_args
        assert kwargs["window_end"] == window_end

    def test_constructs_ingestion_with_injected_config(
        self, base_config: AppConfig
    ) -> None:
        """Ingestion is built from the injected config (path isolation seam)."""
        # Arrange
        window_end = datetime(2021, 1, 18, 0, 15, 0)
        with patch("orchestration.lib.ingest_helpers.Ingestion") as mock_cls:
            mock_cls.return_value.run.return_value = BRONZE_ROOT

            # Act
            run_ingest(window_end, seed_start=SEED_START, config=base_config)

        # Assert
        _, kwargs = mock_cls.call_args
        assert kwargs["config"] is base_config

    def test_returns_bronze_path(self, base_config: AppConfig) -> None:
        """run_ingest returns the path Ingestion.run() reports."""
        # Arrange
        window_end = datetime(2021, 1, 18, 0, 15, 0)
        with patch("orchestration.lib.ingest_helpers.Ingestion") as mock_cls:
            mock_cls.return_value.run.return_value = BRONZE_ROOT

            # Act
            result = run_ingest(window_end, seed_start=SEED_START, config=base_config)

        # Assert
        assert result == BRONZE_ROOT

    def test_run_called_exactly_once(self, base_config: AppConfig) -> None:
        """A single run() call per invocation — no accidental double-ingest."""
        # Arrange
        window_end = datetime(2021, 1, 18, 0, 15, 0)
        with patch("orchestration.lib.ingest_helpers.Ingestion") as mock_cls:
            instance = mock_cls.return_value
            instance.run.return_value = BRONZE_ROOT

            # Act
            run_ingest(window_end, seed_start=SEED_START, config=base_config)

        # Assert
        assert instance.run.call_count == 1
