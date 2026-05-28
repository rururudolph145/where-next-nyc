"""Data cleaning and Bronze-to-Silver promotion for NYC taxi trip data."""

import json
import os
from typing import Any

import pandas as pd

from app.core.paths import validation_log_dir
from app.ingestion.schema import (
    DataValidationError,
    validate_bronze_schema,
    write_validation_log,
)


class DataCleaner:
    """Cleans raw taxi trip data and promotes it from Bronze to Silver layer."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize DataCleaner with pipeline configuration.

        Args:
            config: Parsed contents of config.yml.
        """
        self.config = config
        self.env: str = config["env"]
        self.bronze_path: str = config["data"]["bronze_path"]
        self.silver_path: str = config["data"]["silver_path"]
        self.null_strategy: str = config["clean"]["null_strategy"]

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def load_bronze(self) -> pd.DataFrame:
        """Read raw data from the Bronze layer.

        Returns:
            Raw DataFrame loaded from the Bronze parquet file.

        Raises:
            NotImplementedError: If env is not 'local'.
        """
        if self.env == "local":
            return pd.read_parquet(self.bronze_path)
        raise NotImplementedError(f"env='{self.env}' is not supported yet.")

    def save_silver(self, data: pd.DataFrame) -> None:
        """Write cleaned data to the Silver layer (overwrite).

        Args:
            data: Cleaned DataFrame to persist.

        Raises:
            NotImplementedError: If env is not 'local'.
        """
        if self.env == "local":
            os.makedirs(os.path.dirname(self.silver_path), exist_ok=True)
            data = data.sort_values("tpep_pickup_datetime").reset_index(drop=True)
            data.to_parquet(self.silver_path, index=False)
            print(f"Saved → {self.silver_path}  ({len(data):,} rows)")
            return
        raise NotImplementedError(f"env='{self.env}' is not supported yet.")

    # ------------------------------------------------------------------
    # Null handling
    # ------------------------------------------------------------------

    def handle_nulls(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the configured null-value strategy to the DataFrame.

        When ``config.clean.null_subset`` is provided, only those columns
        are considered for the null check (relevant for ``drop`` strategy).

        Args:
            data: Input DataFrame that may contain null values.

        Returns:
            DataFrame with nulls treated according to config.clean.null_strategy.

        Raises:
            ValueError: If null_strategy is not one of drop, ffill, bfill.
        """
        strategy = self.null_strategy
        subset: list[str] | None = self.config["clean"].get("null_subset")
        if subset:
            # Only keep subset columns that actually exist in the data
            subset = [c for c in subset if c in data.columns]
        if strategy == "drop":
            return data.dropna(subset=subset or None)
        if strategy == "ffill":
            return data.ffill()
        if strategy == "bfill":
            return data.bfill()
        raise ValueError(
            f"Unknown null_strategy '{strategy}'. Choose from: drop, ffill, bfill."
        )

    # ------------------------------------------------------------------
    # Cleaning logic
    # ------------------------------------------------------------------

    def clean(
        self,
        data: pd.DataFrame,
        iqr_bounds: dict[str, float] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """Apply all cleaning and feature-engineering steps.

        Args:
            data: Raw DataFrame from the Bronze layer.
            iqr_bounds: Pre-computed IQR bounds to reuse (transform mode).
                If None, bounds are computed from the data (fit mode).

        Returns:
            Tuple of (cleaned DataFrame, IQR bounds dict with keys
            q1, q3, iqr).
        """
        data = data.copy()

        # Parse datetimes
        data["tpep_pickup_datetime"] = pd.to_datetime(data["tpep_pickup_datetime"])
        data["tpep_dropoff_datetime"] = pd.to_datetime(data["tpep_dropoff_datetime"])

        # Handle nulls with configured strategy
        data = self.handle_nulls(data)

        # Remove trips with non-positive passenger count or trip distance
        data = data[(data["passenger_count"] > 0) & (data["trip_distance"] > 0)]

        # Remove fare outliers using IQR
        if iqr_bounds is None:
            q1 = float(data["fare_amount"].quantile(0.25))
            q3 = float(data["fare_amount"].quantile(0.75))
            iqr = q3 - q1
            bounds: dict[str, float] = {"q1": q1, "q3": q3, "iqr": iqr}
        else:
            bounds = iqr_bounds
            q1 = bounds["q1"]
            q3 = bounds["q3"]
            iqr = bounds["iqr"]

        data = data[
            (data["fare_amount"] >= q1 - 1.5 * iqr)
            & (data["fare_amount"] <= q3 + 1.5 * iqr)
        ]

        # Feature engineering
        data["hour"] = data["tpep_pickup_datetime"].dt.hour
        data["day_of_week"] = data["tpep_pickup_datetime"].dt.dayofweek
        data["month"] = data["tpep_pickup_datetime"].dt.month
        data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

        return data, bounds

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Execute the full Bronze -> Silver cleaning pipeline.

        Splits bronze data by the inference simulation start hour so that
        IQR fare thresholds are fit only on the training period, then
        applied (transformed) to the inference period. This prevents
        data leakage from future data into the cleaning step.

        Returns:
            Cleaned DataFrame that has been saved to the Silver layer.
        """
        print(f"Loading bronze data from: {self.bronze_path}")
        raw = self.load_bronze()
        print(f"  Raw rows: {len(raw):,}")

        # Schema validation — write log regardless of pass/fail
        log_dir = validation_log_dir(self.config)
        failure_reasons: list[str] = []
        try:
            validate_bronze_schema(raw)
        except DataValidationError as exc:
            failure_reasons = exc.failure_reasons
            raise
        finally:
            write_validation_log(log_dir, raw, failure_reasons)

        # Parse pickup datetimes for splitting
        raw["tpep_pickup_datetime"] = pd.to_datetime(raw["tpep_pickup_datetime"])

        # Split by inference cutoff
        cutoff = pd.Timestamp(self.config["inference"]["simulation"]["start_hour"])
        train_raw = raw[raw["tpep_pickup_datetime"] < cutoff]
        inference_raw = raw[raw["tpep_pickup_datetime"] >= cutoff]

        print(f"  Training rows (before {cutoff}): {len(train_raw):,}")
        print(f"  Inference rows (from {cutoff}): {len(inference_raw):,}")

        # Fit IQR on training data
        train_cleaned, bounds = self.clean(train_raw)
        print(f"  Cleaned training rows: {len(train_cleaned):,}")
        print(f"  IQR bounds: {bounds}")

        # Transform inference data with training bounds
        if len(inference_raw) > 0:
            inference_cleaned, _ = self.clean(inference_raw, iqr_bounds=bounds)
            print(f"  Cleaned inference rows: {len(inference_cleaned):,}")
            cleaned = pd.concat([train_cleaned, inference_cleaned], ignore_index=True)
        else:
            cleaned = train_cleaned

        print(f"  Total cleaned rows: {len(cleaned):,}")

        # Persist IQR bounds for reproducibility
        store_path: str = self.config["model"]["store_path"]
        os.makedirs(store_path, exist_ok=True)
        bounds_path = os.path.join(store_path, "iqr_bounds.json")
        with open(bounds_path, "w") as f:
            json.dump(bounds, f, indent=2)
        print(f"  Saved IQR bounds → {bounds_path}")

        self.save_silver(cleaned)
        return cleaned
