"""Tests for orchestration_experiment/dags/exp03b_feature.py.

PBI EXP-03b (experimentation) — the pure helpers the feature DAG delegates to.
Inserts a ``feature`` task between ``validate`` and ``advance_watermark``:
``DataCleaner(config).run()`` (bronze → silver) then
``FeatureEngineer(config).run()`` (silver → gold), behind one task label because
they share the same override config and always run together.

The logic lives in a PURE, Airflow-free module (the experiment container has
Airflow; the host test env does NOT). Two seams keep it unit-testable without
reading a real shapefile or a 340MB bronze:

  * ``build_feature_config`` — deep-overrides ``data.silver_path``,
    ``data.gold_path``, ``feature.zone_shapefile_path``, ``model.store_path``
    (redirect to the experiment root), and ``inference.simulation.start_hour``
    (pin to ``window_end`` so DataCleaner's train/inference split is
    well-defined for the pre-cutoff experiment window — OQ4). Nothing else
    changes.
  * ``run_feature`` — constructs ``DataCleaner(config)``, calls ``.run()``, then
    ``FeatureEngineer(config).run()``, and returns the bronze path unchanged
    (stable XCom flow). We assert order and config injection by mocking both
    classes at the module boundary.

AC coverage:
  - build_feature_config overrides the four experiment paths
  - build_feature_config pins inference.simulation.start_hour to window_end (OQ4)
  - build_feature_config does not mutate the base config (deep copy)
  - build_feature_config preserves unrelated keys
  - run_feature constructs DataCleaner with the injected config and calls run()
  - run_feature constructs FeatureEngineer with the injected config and calls run()
  - run_feature runs the cleaner BEFORE the feature engineer (silver→gold order)
  - run_feature returns the bronze path (stable XCom contract)
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from orchestration_experiment.dags.exp03b_feature import (
    build_feature_config,
    run_feature,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

BRONZE_PATH = "/opt/airflow/data_exp/bronze/yellow_tripdata/bronze.parquet"
SILVER_ROOT = "/opt/airflow/data_exp/silver/yellow_tripdata_cleaned.parquet"
GOLD_ROOT = "/opt/airflow/data_exp/gold/yellow_tripdata"
SHAPEFILE = "/opt/airflow/data/reference/taxi_zones.shp"
MODEL_ROOT = "/opt/airflow/data_exp/models"
WINDOW_END = datetime(2021, 1, 18, 0, 15, 0)


@pytest.fixture
def base_config() -> dict[str, Any]:
    """Minimal config mirroring the shape both classes read."""
    return {
        "env": "experiment",
        "ingest": {"slice_duration": "15min"},
        "clean": {"null_strategy": "drop"},
        "feature": {
            "bucket": "15min",
            "lags": [1, 2],
            "horizon_sum_buckets": 2,
            "n_neighbors": 5,
            "zone_shapefile_path": "data/reference/taxi_zones.shp",
        },
        "data": {
            "bronze_path": "data/bronze/yellow_tripdata/bronze.parquet",
            "silver_path": "data/silver/yellow_tripdata_cleaned.parquet",
            "gold_path": "data/gold/yellow_tripdata",
        },
        "model": {"store_path": "models/"},
        "inference": {"simulation": {"start_hour": "2021-03-01 00:00:00"}},
    }


@pytest.fixture
def override_config(base_config: dict[str, Any]) -> dict[str, Any]:
    """A config already redirected to the experiment roots."""
    return build_feature_config(
        base_config,
        silver_root=SILVER_ROOT,
        gold_root=GOLD_ROOT,
        shapefile_path=SHAPEFILE,
        model_root=MODEL_ROOT,
        window_end=WINDOW_END,
    )


# ---------------------------------------------------------------------------
# TestBuildFeatureConfig
# ---------------------------------------------------------------------------


class TestBuildFeatureConfig:
    """build_feature_config injects the isolated paths + OQ4 cutoff (the seam)."""

    def test_overrides_the_four_paths(self, base_config: dict[str, Any]) -> None:
        """silver/gold/shapefile/model paths point at the experiment roots."""
        # Act
        result = build_feature_config(
            base_config,
            silver_root=SILVER_ROOT,
            gold_root=GOLD_ROOT,
            shapefile_path=SHAPEFILE,
            model_root=MODEL_ROOT,
            window_end=WINDOW_END,
        )

        # Assert
        assert result["data"]["silver_path"] == SILVER_ROOT
        assert result["data"]["gold_path"] == GOLD_ROOT
        assert result["feature"]["zone_shapefile_path"] == SHAPEFILE
        assert result["model"]["store_path"] == MODEL_ROOT

    def test_pins_start_hour_to_window_end(self, base_config: dict[str, Any]) -> None:
        """OQ4: the train/inference cutoff is pinned to window_end."""
        # Act
        result = build_feature_config(
            base_config,
            silver_root=SILVER_ROOT,
            gold_root=GOLD_ROOT,
            shapefile_path=SHAPEFILE,
            model_root=MODEL_ROOT,
            window_end=WINDOW_END,
        )

        # Assert
        assert result["inference"]["simulation"]["start_hour"] == (
            WINDOW_END.isoformat()
        )

    def test_does_not_mutate_base(self, base_config: dict[str, Any]) -> None:
        """The base config is deep-copied, never mutated in place."""
        # Arrange
        snapshot = copy.deepcopy(base_config)

        # Act
        build_feature_config(
            base_config,
            silver_root=SILVER_ROOT,
            gold_root=GOLD_ROOT,
            shapefile_path=SHAPEFILE,
            model_root=MODEL_ROOT,
            window_end=WINDOW_END,
        )

        # Assert
        assert base_config == snapshot

    def test_preserves_unrelated_keys(self, base_config: dict[str, Any]) -> None:
        """Only the redirected keys change; everything else passes through."""
        # Act
        result = build_feature_config(
            base_config,
            silver_root=SILVER_ROOT,
            gold_root=GOLD_ROOT,
            shapefile_path=SHAPEFILE,
            model_root=MODEL_ROOT,
            window_end=WINDOW_END,
        )

        # Assert
        assert result["data"]["bronze_path"] == base_config["data"]["bronze_path"]
        assert result["clean"] == base_config["clean"]
        assert result["feature"]["lags"] == base_config["feature"]["lags"]
        assert (
            result["feature"]["horizon_sum_buckets"]
            == base_config["feature"]["horizon_sum_buckets"]
        )


# ---------------------------------------------------------------------------
# TestRunFeature
# ---------------------------------------------------------------------------


class TestRunFeature:
    """run_feature wires DataCleaner → FeatureEngineer with the override config."""

    def test_constructs_cleaner_with_injected_config(
        self, override_config: dict[str, Any]
    ) -> None:
        """DataCleaner is built with the override config and run() is called."""
        # Arrange
        with (
            patch(
                "orchestration_experiment.dags.exp03b_feature.DataCleaner"
            ) as mock_cleaner,
            patch("orchestration_experiment.dags.exp03b_feature.FeatureEngineer"),
        ):
            # Act
            run_feature(BRONZE_PATH, config=override_config)

        # Assert
        mock_cleaner.assert_called_once_with(override_config)
        mock_cleaner.return_value.run.assert_called_once_with()

    def test_constructs_engineer_with_injected_config(
        self, override_config: dict[str, Any]
    ) -> None:
        """FeatureEngineer is built with the override config and run() is called."""
        # Arrange
        with (
            patch("orchestration_experiment.dags.exp03b_feature.DataCleaner"),
            patch(
                "orchestration_experiment.dags.exp03b_feature.FeatureEngineer"
            ) as mock_engineer,
        ):
            # Act
            run_feature(BRONZE_PATH, config=override_config)

        # Assert
        mock_engineer.assert_called_once_with(config=override_config)
        mock_engineer.return_value.run.assert_called_once_with()

    def test_cleaner_runs_before_engineer(
        self, override_config: dict[str, Any]
    ) -> None:
        """Silver must exist before gold: cleaner.run() precedes engineer.run()."""
        # Arrange
        manager = MagicMock()
        with (
            patch(
                "orchestration_experiment.dags.exp03b_feature.DataCleaner",
                manager.DataCleaner,
            ),
            patch(
                "orchestration_experiment.dags.exp03b_feature.FeatureEngineer",
                manager.FeatureEngineer,
            ),
        ):
            # Act
            run_feature(BRONZE_PATH, config=override_config)

        # Assert — the cleaner's run() call is recorded before the engineer's
        call_names = [c[0] for c in manager.mock_calls]
        cleaner_run = call_names.index("DataCleaner().run")
        engineer_run = call_names.index("FeatureEngineer().run")
        assert cleaner_run < engineer_run

    def test_returns_bronze_path(self, override_config: dict[str, Any]) -> None:
        """A stable XCom contract: run_feature hands the bronze path forward."""
        # Arrange
        with (
            patch("orchestration_experiment.dags.exp03b_feature.DataCleaner"),
            patch("orchestration_experiment.dags.exp03b_feature.FeatureEngineer"),
        ):
            # Act
            result = run_feature(BRONZE_PATH, config=override_config)

        # Assert
        assert result == BRONZE_PATH
