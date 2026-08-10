"""Centralized configuration loading for the NYC Taxi Demand Forecasting project."""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class IngestConfig(BaseModel):
    dataset_max: str
    zone_id_min: int
    zone_id_max: int
    start_year_month: str
    end_year_month: str
    max_workers: int = 8  # source: ingest.py:43 fallback
    seed_start: str
    slice_duration: str
    bronze_schema: dict[str, str]


class DataConfig(BaseModel):
    raw_path: str
    train_path: str
    test_path: str
    bronze_path: str
    silver_path: str
    gold_path: str


class FeatureConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    bucket: str
    lags: list[int] = Field(validation_alias=AliasChoices("lags", "lag_hours"))
    horizon_buckets: int = 0  # source: feature.py:82 fallback
    horizon_sum_buckets: int = (
        0  # source: feature.py:88 fallback (config.yml=2, intentional mismatch)
    )
    n_neighbors: int = (
        0  # source: feature.py:92 fallback (config.yml=5, intentional mismatch)
    )
    zone_shapefile_path: str


class CleanConfig(BaseModel):
    iqr_multiplier: float = 1.5  # source: clean.py:144 fallback
    null_strategy: str
    null_subset: list[str] | None = None  # source: clean.py:86 .get()→None


class TrainGateConfig(BaseModel):
    min_wape_lift_pct: float = 2.0  # source: trainer.py:71 fallback


class TrainConfig(BaseModel):
    window_days: int
    gate: TrainGateConfig
    split_ratios: list[float]
    features: list[str]
    target: str


class ModelConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(
        default_factory=dict
    )  # source: trainer.py:50 fallback
    store_path: str


class InferenceSimConfig(BaseModel):
    start_hour: str
    end_hour: str


class InferenceConfig(BaseModel):
    output_path: str = "data/predictions/demand_predictions.parquet"  # source: predictor.py:48 fallback
    simulation: InferenceSimConfig


class ServingConfig(BaseModel):
    watermark_path: str
    predictions_path: str
    zone_shapefile_path: str


class MonitoringConfig(BaseModel):
    predictions_path: str = (
        "data/monitoring/predictions"  # source: actuals_join.py:189 fallback
    )
    joined_path: str = "data/monitoring/joined"  # source: actuals_join.py:193 fallback


class MlflowConfig(BaseModel):
    experiment_name: str = "ny_taxi_demand_model"  # source: trainer.py:62 fallback
    tracking_uri: str = (
        "sqlite:///mlflow.db"  # source: predictor.py:53, trainer.py:65 fallback
    )
    model_name: str = (
        "ny_taxi_demand_model"  # source: predictor.py:55, trainer.py:66 fallback
    )


class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    env: str
    ingest: IngestConfig
    data: DataConfig
    feature: FeatureConfig
    clean: CleanConfig
    train: TrainConfig
    model: ModelConfig
    inference: InferenceConfig
    serving: ServingConfig
    monitoring: MonitoringConfig
    mlflow: MlflowConfig


def load_config(path: str = "config.yml") -> AppConfig:
    """Load and parse the YAML configuration file into a typed AppConfig.

    Args:
        path: Path to the YAML configuration file. Defaults to
            ``"config.yml"`` relative to the current working directory.

    Returns:
        Parsed, validated AppConfig. Raises pydantic.ValidationError if the
        file is missing required fields or contains type-wrong values.
    """
    with open(path) as f:
        return AppConfig.model_validate(yaml.safe_load(f))
