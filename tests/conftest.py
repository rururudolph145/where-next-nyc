"""Shared pytest fixtures and helpers for all test modules."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.config import AppConfig

# ---------------------------------------------------------------------------
# Canonical full-config dict — mirrors config.yml exactly.
# Each vertical-slice test overrides only its own fields via
# full_config.model_copy(update={...}) (nested overrides require sub-model copy).
# ---------------------------------------------------------------------------
_FULL_CONFIG_DICT: dict = {
    "env": "local",
    "ingest": {
        "dataset_max": "2022-03-31T23:59:57",
        "zone_id_min": 1,
        "zone_id_max": 265,
        "start_year_month": "2021-01",
        "end_year_month": "2022-03",
        "max_workers": 8,
        "seed_start": "2021-01-18T00:00:00",
        "slice_duration": "15min",
        "bronze_schema": {
            "VendorID": "float64",
            "tpep_pickup_datetime": "datetime64[us]",
            "tpep_dropoff_datetime": "datetime64[us]",
            "passenger_count": "float64",
            "trip_distance": "float64",
            "RatecodeID": "float64",
            "store_and_fwd_flag": "str",
            "PULocationID": "float64",
            "DOLocationID": "float64",
            "payment_type": "float64",
            "fare_amount": "float64",
            "extra": "float64",
            "mta_tax": "float64",
            "tip_amount": "float64",
            "tolls_amount": "float64",
            "improvement_surcharge": "float64",
            "total_amount": "float64",
            "congestion_surcharge": "float64",
            "airport_fee": "float64",
        },
    },
    "data": {
        "raw_path": "data/raw/yellow_tripdata",
        "train_path": "data/train.parquet",
        "test_path": "data/test.parquet",
        "bronze_path": "data/bronze/yellow_tripdata/bronze.parquet",
        "silver_path": "data/silver/yellow_tripdata_cleaned.parquet",
        "gold_path": "data/gold/yellow_tripdata",
    },
    "feature": {
        "bucket": "15min",
        "lags": [1, 2, 4, 8, 12, 24, 96, 672],
        "horizon_buckets": 0,
        "horizon_sum_buckets": 2,
        "n_neighbors": 5,
        "zone_shapefile_path": "data/reference/taxi_zones.shp",
    },
    "clean": {
        "iqr_multiplier": 1.5,
        "null_strategy": "drop",
        "null_subset": [
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "passenger_count",
            "trip_distance",
            "fare_amount",
            "PULocationID",
            "DOLocationID",
        ],
    },
    "train": {
        "window_days": 30,
        "gate": {"min_wape_lift_pct": 2.0},
        "features": [
            "hour_of_day_sin",
            "hour_of_day_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "month_sin",
            "month_cos",
            "pickup_longitude",
            "pickup_latitude",
            "trip_count_lag_1",
            "trip_count_lag_2",
            "trip_count_lag_4",
            "trip_count_lag_8",
            "trip_count_lag_12",
            "trip_count_lag_24",
            "trip_count_lag_96",
            "trip_count_lag_672",
            "trip_count_lag_1_neighbor_mean",
        ],
        "target": "target_sum_t1_t2",
    },
    "model": {
        "name": "LGBMRegressor",
        "params": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": -1,
            "num_leaves": 31,
            "random_state": 42,
            "verbosity": -1,
        },
        "store_path": "data/models",
    },
    "inference": {
        "output_path": "data/predictions/demand_predictions.parquet",
        "simulation": {
            "start_hour": "2021-03-01 00:00:00",
            "end_hour": "2021-03-02 23:00:00",
        },
    },
    "serving": {
        "watermark_path": "data/watermarks/refresh_15min",
        "predictions_path": "data/predictions/demand_predictions.parquet",
        "zone_shapefile_path": "data/reference/taxi_zones.shp",
    },
    "monitoring": {
        "predictions_path": "data/monitoring/predictions",
        "joined_path": "data/monitoring/joined",
    },
    "mlflow": {
        "experiment_name": "ny_taxi_demand_model",
        "tracking_uri": "sqlite:///data/mlflow.db",
        "model_name": "ny_taxi_demand_model",
    },
}


@pytest.fixture()
def full_config() -> AppConfig:
    """Return an AppConfig built from the canonical _FULL_CONFIG_DICT.

    Each test overrides only its own fields:
        full_config.model_copy(update={"data": full_config.data.model_copy(update={"raw_path": "x"})})
    """
    return AppConfig.model_validate(_FULL_CONFIG_DICT)


def make_mock_version(
    version: str = "3",
    stage: str = "Production",
    creation_timestamp_ms: int = 1_710_058_921_000,
    tags: dict[str, str] | None = None,
) -> MagicMock:
    """Return a MagicMock simulating an MLflow ModelVersion.

    Shared across test modules that exercise MlflowClient.get_latest_versions().

    Args:
        version: The model version string (e.g. "3").
        stage: The MLflow stage string ("Production" or "Archived").
        creation_timestamp_ms: Unix epoch milliseconds for the version's creation time.
        tags: Optional dict of MLflow model version tags.

    Returns:
        MagicMock with ``.version``, ``.current_stage``, ``.creation_timestamp``,
        and ``.tags`` attributes set.
    """
    mv = MagicMock()
    mv.version = version
    mv.current_stage = stage
    mv.creation_timestamp = creation_timestamp_ms
    mv.tags = tags if tags is not None else {}
    return mv
