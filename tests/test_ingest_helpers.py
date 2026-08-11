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
from pathlib import Path
from unittest.mock import patch

import pandas as pd
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
# TestRunIngest  (updated contract: incremental append, not cumulative rewrite)
# ---------------------------------------------------------------------------

WINDOW_START = datetime(2021, 1, 18, 0, 15, 0)


class TestRunIngest:
    """run_ingest — incremental branch when bronze exists, full seed when absent."""

    # --- kept from original contract (still valid) ---

    def test_constructs_ingestion_with_injected_config(
        self, base_config: AppConfig
    ) -> None:
        """Ingestion is built from the injected config (path isolation seam)."""
        # Arrange
        window_end = datetime(2021, 1, 18, 0, 30, 0)
        with patch("orchestration.lib.ingest_helpers.Ingestion") as mock_cls:
            instance = mock_cls.return_value
            instance.bronze_exists.return_value = False
            instance.load_raw.return_value = pd.DataFrame(
                {"tpep_pickup_datetime": pd.to_datetime(["2021-01-18"])}
            )
            instance._filter_window.return_value = instance.load_raw.return_value
            instance.save_bronze.return_value = BRONZE_ROOT

            # Act
            run_ingest(
                window_end,
                window_start=WINDOW_START,
                seed_start=SEED_START,
                config=base_config,
            )

        # Assert
        _, kwargs = mock_cls.call_args
        assert kwargs["config"] is base_config

    def test_returns_bronze_path(self, base_config: AppConfig) -> None:
        """run_ingest returns the path save_bronze() reports."""
        # Arrange
        window_end = datetime(2021, 1, 18, 0, 30, 0)
        with patch("orchestration.lib.ingest_helpers.Ingestion") as mock_cls:
            instance = mock_cls.return_value
            instance.bronze_exists.return_value = False
            instance.load_raw.return_value = pd.DataFrame(
                {"tpep_pickup_datetime": pd.to_datetime(["2021-01-18"])}
            )
            instance._filter_window.return_value = instance.load_raw.return_value
            instance.save_bronze.return_value = BRONZE_ROOT

            # Act
            result = run_ingest(
                window_end,
                window_start=WINDOW_START,
                seed_start=SEED_START,
                config=base_config,
            )

        # Assert
        assert result == BRONZE_ROOT

    # --- Step 8: first-run seeds full history ---

    def test_run_ingest_first_run_seeds_full_history(
        self, base_config: AppConfig
    ) -> None:
        """First run (no bronze): load_raw called with since=None, save_bronze once.

        AC: full [seed_start, window_end) seed on first tick.
        """
        # Arrange
        window_end = datetime(2021, 1, 18, 0, 30, 0)
        with patch("orchestration.lib.ingest_helpers.Ingestion") as mock_cls:
            instance = mock_cls.return_value
            instance.bronze_exists.return_value = False
            instance.load_raw.return_value = pd.DataFrame(
                {"tpep_pickup_datetime": pd.to_datetime(["2021-01-18"])}
            )
            instance._filter_window.return_value = instance.load_raw.return_value
            instance.save_bronze.return_value = BRONZE_ROOT

            # Act
            result = run_ingest(
                window_end,
                window_start=WINDOW_START,
                seed_start=SEED_START,
                config=base_config,
            )

        # Assert
        instance.load_raw.assert_called_once_with(since=None)
        instance.save_bronze.assert_called_once()
        assert result == BRONZE_ROOT

    # --- Step 9: incremental reads only window month ---

    def test_run_ingest_incremental_reads_only_window_month(
        self, base_config: AppConfig
    ) -> None:
        """Incremental (bronze exists): load_raw called with since=window_start.

        AC: only the new slice's month is read, not the full 6-month history.
        """
        # Arrange
        window_end = datetime(2021, 1, 18, 0, 30, 0)
        with patch("orchestration.lib.ingest_helpers.Ingestion") as mock_cls:
            instance = mock_cls.return_value
            instance.bronze_exists.return_value = True
            existing = pd.DataFrame(
                {"tpep_pickup_datetime": pd.to_datetime(["2021-01-18 00:00"])}
            )
            new_slice = pd.DataFrame(
                {"tpep_pickup_datetime": pd.to_datetime(["2021-01-18 00:15"])}
            )
            instance.load_bronze.return_value = existing
            instance.load_raw.return_value = new_slice
            instance._filter_window.return_value = new_slice
            instance.append_slice.return_value = pd.concat(
                [existing, new_slice], ignore_index=True
            )
            instance.save_bronze.return_value = BRONZE_ROOT

            # Act
            run_ingest(
                window_end,
                window_start=WINDOW_START,
                seed_start=SEED_START,
                config=base_config,
            )

        # Assert — since=window_start, NOT None
        instance.load_raw.assert_called_once_with(since=WINDOW_START)

    # --- Step 10: incremental appends with correct cutoff ---

    def test_run_ingest_incremental_appends_with_cutoff(
        self, base_config: AppConfig
    ) -> None:
        """Incremental path calls append_slice with cutoff == window_start.

        AC: the idempotent replace boundary is window_start (not seed_start).
        """
        # Arrange
        window_end = datetime(2021, 1, 18, 0, 30, 0)
        with patch("orchestration.lib.ingest_helpers.Ingestion") as mock_cls:
            instance = mock_cls.return_value
            instance.bronze_exists.return_value = True
            existing = pd.DataFrame(
                {"tpep_pickup_datetime": pd.to_datetime(["2021-01-18 00:00"])}
            )
            new_slice = pd.DataFrame(
                {"tpep_pickup_datetime": pd.to_datetime(["2021-01-18 00:15"])}
            )
            instance.load_bronze.return_value = existing
            instance.load_raw.return_value = new_slice
            instance._filter_window.return_value = new_slice
            instance.append_slice.return_value = pd.concat(
                [existing, new_slice], ignore_index=True
            )
            instance.save_bronze.return_value = BRONZE_ROOT

            # Act
            run_ingest(
                window_end,
                window_start=WINDOW_START,
                seed_start=SEED_START,
                config=base_config,
            )

        # Assert
        _, kwargs = instance.append_slice.call_args
        assert kwargs["cutoff"] == WINDOW_START

    # --- Step 10 / save invariant: save_bronze called exactly once per branch ---

    def test_save_bronze_called_once_per_tick(self, base_config: AppConfig) -> None:
        """save_bronze is called exactly once per run_ingest invocation.

        AC: single parquet overwrite — no accidental double-write.
        """
        # Arrange
        window_end = datetime(2021, 1, 18, 0, 30, 0)
        with patch("orchestration.lib.ingest_helpers.Ingestion") as mock_cls:
            instance = mock_cls.return_value
            instance.bronze_exists.return_value = True
            existing = pd.DataFrame(
                {"tpep_pickup_datetime": pd.to_datetime(["2021-01-18 00:00"])}
            )
            new_slice = pd.DataFrame(
                {"tpep_pickup_datetime": pd.to_datetime(["2021-01-18 00:15"])}
            )
            instance.load_bronze.return_value = existing
            instance.load_raw.return_value = new_slice
            instance._filter_window.return_value = new_slice
            instance.append_slice.return_value = pd.concat(
                [existing, new_slice], ignore_index=True
            )
            instance.save_bronze.return_value = BRONZE_ROOT

            # Act
            run_ingest(
                window_end,
                window_start=WINDOW_START,
                seed_start=SEED_START,
                config=base_config,
            )

        # Assert
        assert instance.save_bronze.call_count == 1

    # --- Step 8 (window_end still forwarded via _filter_window) ---

    def test_passes_window_end_through(self, base_config: AppConfig) -> None:
        """window_end is forwarded to _filter_window on the first-run path.

        AC: the full-seed window is correctly bounded at the advancing end.
        """
        # Arrange
        window_end = datetime(2021, 1, 18, 0, 30, 0)
        with patch("orchestration.lib.ingest_helpers.Ingestion") as mock_cls:
            instance = mock_cls.return_value
            instance.bronze_exists.return_value = False
            instance.load_raw.return_value = pd.DataFrame(
                {"tpep_pickup_datetime": pd.to_datetime(["2021-01-18"])}
            )
            instance._filter_window.return_value = instance.load_raw.return_value
            instance.save_bronze.return_value = BRONZE_ROOT

            # Act
            run_ingest(
                window_end,
                window_start=WINDOW_START,
                seed_start=SEED_START,
                config=base_config,
            )

        # Assert — window_end is the exclusive upper bound passed to _filter_window
        args, _ = instance._filter_window.call_args
        assert args[2] == window_end


# ---------------------------------------------------------------------------
# Step 11: capstone — real Ingestion round-trip idempotency (tmp_path, tiny data)
# ---------------------------------------------------------------------------


def _write_raw_parquet(path: Path, timestamps: list[datetime]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "VendorID": [1.0] * len(timestamps),
            "tpep_pickup_datetime": pd.to_datetime(timestamps),
            "tpep_dropoff_datetime": pd.to_datetime(timestamps),
            "passenger_count": [1.0] * len(timestamps),
            "trip_distance": [1.0] * len(timestamps),
            "PULocationID": [1.0] * len(timestamps),
            "DOLocationID": [2.0] * len(timestamps),
            "fare_amount": [5.0] * len(timestamps),
        }
    )
    df.to_parquet(path, index=False)


_BRONZE_SCHEMA = {
    "VendorID": "float64",
    "tpep_pickup_datetime": "datetime64[us]",
    "tpep_dropoff_datetime": "datetime64[us]",
    "passenger_count": "float64",
    "trip_distance": "float64",
    "PULocationID": "float64",
    "DOLocationID": "float64",
    "fare_amount": "float64",
}


class TestRunIngestEndToEnd:
    """Step 11: real Ingestion capstone — idempotency on duplicate window run."""

    def _make_config(self, tmp_path: Path) -> AppConfig:
        from tests.conftest import _FULL_CONFIG_DICT

        base = AppConfig.model_validate(_FULL_CONFIG_DICT)
        raw_root = tmp_path / "raw"
        bronze_path = tmp_path / "bronze" / "bronze.parquet"
        return base.model_copy(
            update={
                "data": base.data.model_copy(
                    update={
                        "raw_path": str(raw_root),
                        "bronze_path": str(bronze_path),
                    }
                ),
                "ingest": base.ingest.model_copy(
                    update={
                        "start_year_month": "2021-01",
                        "end_year_month": "2021-01",
                        "bronze_schema": _BRONZE_SCHEMA,
                    }
                ),
            }
        )

    def test_run_ingest_incremental_idempotent_end_to_end(self, tmp_path: Path) -> None:
        """Seed bronze, run window W, run same window W again — rowcount unchanged.

        AC: retry-safe. A DAG retry replaying the same window does not grow bronze.
        """
        # Arrange — two 15-min windows of tiny raw data in 2021-01
        seed_start = datetime(2021, 1, 18, 0, 0, 0)
        window_start = datetime(2021, 1, 18, 0, 15, 0)
        window_end = datetime(2021, 1, 18, 0, 30, 0)

        config = self._make_config(tmp_path)
        raw_root = Path(config.data.raw_path)

        # Seed rows: one row in the seed window [seed_start, window_start)
        _write_raw_parquet(
            raw_root / "2021_01" / "2021_01_18_seed.parquet",
            [seed_start],
        )
        # New-slice rows: one row in window W
        _write_raw_parquet(
            raw_root / "2021_01" / "2021_01_18_new.parquet",
            [window_start],
        )

        # Act — first run: seed (no bronze yet)
        run_ingest(
            window_end,
            window_start=window_start,
            seed_start=seed_start,
            config=config,
        )
        bronze_after_first = pd.read_parquet(config.data.bronze_path)
        count_first = len(bronze_after_first)

        # Act — second run: same window W (idempotent retry)
        run_ingest(
            window_end,
            window_start=window_start,
            seed_start=seed_start,
            config=config,
        )
        bronze_after_second = pd.read_parquet(config.data.bronze_path)

        # Assert — rowcount must not grow on the retry
        assert len(bronze_after_second) == count_first


# ---------------------------------------------------------------------------
# Step 12: DAG wiring — _ingest pulls window_start from XCom and forwards it
# ---------------------------------------------------------------------------


class TestIngestDagTaskWiring:
    """_ingest callable forwards window_start from read_watermark XCom to run_ingest."""

    def test_ingest_forwards_window_start(self, base_config: AppConfig) -> None:
        """_ingest passes window_start kwarg to run_ingest (not just window_end).

        AC: the DAG task correctly wires the incremental lower bound.
        """
        # Arrange
        pytest.importorskip(
            "airflow", reason="Airflow only present inside the container"
        )
        from unittest.mock import MagicMock

        from orchestration.dags.the_15_minute_refresh_inference_dag import _ingest

        window = {
            "window_start": "2021-01-18T00:15:00",
            "window_end": "2021-01-18T00:30:00",
        }
        ti_mock = MagicMock()
        ti_mock.xcom_pull.return_value = window

        with (
            patch(
                "orchestration.dags.the_15_minute_refresh_inference_dag.run_ingest"
            ) as mock_run_ingest,
            patch(
                "orchestration.dags.the_15_minute_refresh_inference_dag.load_config"
            ) as mock_load_config,
            patch(
                "orchestration.dags.the_15_minute_refresh_inference_dag.container_paths"
            ) as mock_paths,
            patch(
                "orchestration.dags.the_15_minute_refresh_inference_dag.build_ingest_config"
            ) as mock_build,
        ):
            mock_load_config.return_value = base_config
            mock_paths.return_value = {
                "raw": "/raw",
                "bronze": "/bronze/bronze.parquet",
            }
            mock_build.return_value = base_config
            mock_run_ingest.return_value = "/bronze/bronze.parquet"

            # Act
            _ingest(ti=ti_mock)

        # Assert — window_start forwarded as a kwarg
        _, kwargs = mock_run_ingest.call_args
        assert kwargs["window_start"] == datetime(2021, 1, 18, 0, 15, 0)
