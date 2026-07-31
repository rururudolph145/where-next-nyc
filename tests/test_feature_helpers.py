"""Tests for orchestration/lib/feature_helpers.py.

The pure, Airflow-free helpers the ``feature`` task of
``the_15_minute_refresh_inference_dag`` delegates to. Cleans bronze → silver
then engineers silver → gold, behind one task label because both steps share
the same override config and always run together.

Two seams keep it unit-testable without a real shapefile or a 340MB bronze:

  * ``build_feature_config`` — deep-overrides ``data.silver_path``,
    ``data.gold_path``, ``feature.zone_shapefile_path`` and ``model.store_path``
    (redirect to the isolated roots). Nothing else changes, and the base is
    never mutated.
  * ``run_feature`` — TRANSFORM-ONLY cleaning: it constructs
    ``DataCleaner(config)``, loads bronze, reads pre-fitted IQR bounds from
    ``model.store_path/iqr_bounds.json``, calls ``clean(raw, iqr_bounds=...)``
    (NOT ``run()`` — no re-split, no re-fit on the cumulative bronze), saves
    silver, then runs ``FeatureEngineer(config=config).run()``. Returns the
    bronze path unchanged (stable XCom flow).

AC coverage:
  - build_feature_config overrides the four isolated paths
  - build_feature_config does not mutate the base config (deep copy)
  - build_feature_config preserves unrelated keys
  - run_feature reads IQR bounds from model.store_path/iqr_bounds.json
  - run_feature cleans transform-only: clean(iqr_bounds=...) is called and
    DataCleaner.run() is NEVER called (anti-regression on re-fitting)
  - run_feature saves the cleaned frame to silver
  - run_feature constructs FeatureEngineer with the injected config and runs it
  - run_feature cleans BEFORE engineering (silver→gold order)
  - run_feature returns the bronze path (stable XCom contract)
"""

from __future__ import annotations

import copy
import json
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

import pytest

from orchestration.lib.feature_helpers import (
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
IQR_BOUNDS = {"trip_distance": [0.0, 25.0], "fare_amount": [0.0, 100.0]}


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
    """A config already redirected to the isolated roots."""
    return build_feature_config(
        base_config,
        silver_root=SILVER_ROOT,
        gold_root=GOLD_ROOT,
        shapefile_path=SHAPEFILE,
        model_root=MODEL_ROOT,
    )


# ---------------------------------------------------------------------------
# TestBuildFeatureConfig
# ---------------------------------------------------------------------------


class TestBuildFeatureConfig:
    """build_feature_config injects the isolated paths (the seam)."""

    def test_overrides_the_four_paths(self, base_config: dict[str, Any]) -> None:
        """silver/gold/shapefile/model paths point at the isolated roots."""
        # Act
        result = build_feature_config(
            base_config,
            silver_root=SILVER_ROOT,
            gold_root=GOLD_ROOT,
            shapefile_path=SHAPEFILE,
            model_root=MODEL_ROOT,
        )

        # Assert
        assert result["data"]["silver_path"] == SILVER_ROOT
        assert result["data"]["gold_path"] == GOLD_ROOT
        assert result["feature"]["zone_shapefile_path"] == SHAPEFILE
        assert result["model"]["store_path"] == MODEL_ROOT

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
    """run_feature wires DataCleaner (transform-only) → FeatureEngineer."""

    def test_reads_iqr_bounds_from_model_store_path(
        self, override_config: dict[str, Any]
    ) -> None:
        """IQR bounds are read from ``model.store_path/iqr_bounds.json``."""
        # Arrange
        m = mock_open(read_data=json.dumps(IQR_BOUNDS))
        with (
            patch("orchestration.lib.feature_helpers.DataCleaner") as mock_cleaner_cls,
            patch("orchestration.lib.feature_helpers.FeatureEngineer"),
            patch("orchestration.lib.feature_helpers.open", m),
        ):
            mock_cleaner_cls.return_value.clean.return_value = ("CLEANED_DF", {})
            # Act
            run_feature(BRONZE_PATH, config=override_config)

        # Assert — opened the JSON under the (overridden) model store path
        opened_path = m.call_args[0][0]
        assert opened_path == f"{MODEL_ROOT}/iqr_bounds.json"

    def test_cleans_transform_only_and_never_calls_run(
        self, override_config: dict[str, Any]
    ) -> None:
        """Anti-regression: clean() runs with pre-fitted bounds; run() never fires.

        The whole point of the production rewrite: cleaning must be
        transform-only (reuse training-time IQR bounds), never
        ``DataCleaner(config).run()`` which would re-split and re-fit on the
        cumulative bronze.
        """
        # Arrange
        m = mock_open(read_data=json.dumps(IQR_BOUNDS))
        with (
            patch("orchestration.lib.feature_helpers.DataCleaner") as mock_cleaner_cls,
            patch("orchestration.lib.feature_helpers.FeatureEngineer"),
            patch("orchestration.lib.feature_helpers.open", m),
        ):
            cleaner = mock_cleaner_cls.return_value
            cleaner.clean.return_value = ("CLEANED_DF", IQR_BOUNDS)

            # Act
            run_feature(BRONZE_PATH, config=override_config)

        # Assert
        mock_cleaner_cls.assert_called_once_with(override_config)
        cleaner.clean.assert_called_once_with(
            cleaner.load_bronze.return_value, iqr_bounds=IQR_BOUNDS
        )
        cleaner.run.assert_not_called()

    def test_saves_cleaned_frame_to_silver(
        self, override_config: dict[str, Any]
    ) -> None:
        """The frame returned by clean() is handed to save_silver()."""
        # Arrange
        m = mock_open(read_data=json.dumps(IQR_BOUNDS))
        with (
            patch("orchestration.lib.feature_helpers.DataCleaner") as mock_cleaner_cls,
            patch("orchestration.lib.feature_helpers.FeatureEngineer"),
            patch("orchestration.lib.feature_helpers.open", m),
        ):
            cleaner = mock_cleaner_cls.return_value
            cleaner.clean.return_value = ("CLEANED_DF", IQR_BOUNDS)

            # Act
            run_feature(BRONZE_PATH, config=override_config)

        # Assert
        cleaner.save_silver.assert_called_once_with("CLEANED_DF")

    def test_constructs_engineer_with_injected_config(
        self, override_config: dict[str, Any]
    ) -> None:
        """FeatureEngineer is built with the override config and run() is called."""
        # Arrange
        m = mock_open(read_data=json.dumps(IQR_BOUNDS))
        with (
            patch("orchestration.lib.feature_helpers.DataCleaner") as mock_cleaner_cls,
            patch("orchestration.lib.feature_helpers.FeatureEngineer") as mock_engineer,
            patch("orchestration.lib.feature_helpers.open", m),
        ):
            mock_cleaner_cls.return_value.clean.return_value = ("CLEANED_DF", {})
            # Act
            run_feature(BRONZE_PATH, config=override_config)

        # Assert
        mock_engineer.assert_called_once_with(config=override_config)
        mock_engineer.return_value.run.assert_called_once_with()

    def test_cleaner_saves_before_engineer_runs(
        self, override_config: dict[str, Any]
    ) -> None:
        """Silver must exist before gold: save_silver() precedes engineer.run()."""
        # Arrange
        manager = MagicMock()
        manager.DataCleaner.return_value.clean.return_value = ("CLEANED_DF", {})
        m = mock_open(read_data=json.dumps(IQR_BOUNDS))
        with (
            patch(
                "orchestration.lib.feature_helpers.DataCleaner",
                manager.DataCleaner,
            ),
            patch(
                "orchestration.lib.feature_helpers.FeatureEngineer",
                manager.FeatureEngineer,
            ),
            patch("orchestration.lib.feature_helpers.open", m),
        ):
            # Act
            run_feature(BRONZE_PATH, config=override_config)

        # Assert — save_silver is recorded before the engineer's run()
        call_names = [c[0] for c in manager.mock_calls]
        cleaner_save = call_names.index("DataCleaner().save_silver")
        engineer_run = call_names.index("FeatureEngineer().run")
        assert cleaner_save < engineer_run

    def test_returns_bronze_path(self, override_config: dict[str, Any]) -> None:
        """A stable XCom contract: run_feature hands the bronze path forward."""
        # Arrange
        m = mock_open(read_data=json.dumps(IQR_BOUNDS))
        with (
            patch("orchestration.lib.feature_helpers.DataCleaner") as mock_cleaner_cls,
            patch("orchestration.lib.feature_helpers.FeatureEngineer"),
            patch("orchestration.lib.feature_helpers.open", m),
        ):
            mock_cleaner_cls.return_value.clean.return_value = ("CLEANED_DF", {})
            # Act
            result = run_feature(BRONZE_PATH, config=override_config)

        # Assert
        assert result == BRONZE_PATH
