"""Data cleaning and Bronze-to-Silver promotion for NYC taxi trip data."""

import os
from typing import Any

import pandas as pd


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
        """Upsert cleaned data into the Silver layer.

        If the silver file already exists, the new data is appended and
        duplicates are dropped so the file always contains a single
        deduplicated copy of all ingested records.

        Args:
            data: Cleaned DataFrame to upsert.

        Raises:
            NotImplementedError: If env is not 'local'.
        """
        if self.env == "local":
            os.makedirs(os.path.dirname(self.silver_path), exist_ok=True)

            if os.path.exists(self.silver_path):
                existing = pd.read_parquet(self.silver_path)
                data = pd.concat([existing, data], ignore_index=True)
                data = data.drop_duplicates()

            data = data.sort_values("tpep_pickup_datetime").reset_index(
                drop=True
            )
            data.to_parquet(self.silver_path, index=False)
            print(f"Saved → {self.silver_path}  ({len(data):,} rows)")
            return
        raise NotImplementedError(f"env='{self.env}' is not supported yet.")

    # ------------------------------------------------------------------
    # Null handling
    # ------------------------------------------------------------------

    def handle_nulls(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the configured null-value strategy to the DataFrame.

        Args:
            data: Input DataFrame that may contain null values.

        Returns:
            DataFrame with nulls treated according to config.clean.null_strategy.

        Raises:
            ValueError: If null_strategy is not one of drop, ffill, bfill.
        """
        strategy = self.null_strategy
        if strategy == "drop":
            return data.dropna()
        if strategy == "ffill":
            return data.ffill()
        if strategy == "bfill":
            return data.bfill()
        raise ValueError(
            f"Unknown null_strategy '{strategy}'. "
            "Choose from: drop, ffill, bfill."
        )

    # ------------------------------------------------------------------
    # Cleaning logic
    # ------------------------------------------------------------------

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all cleaning and feature-engineering steps.

        Args:
            data: Raw DataFrame from the Bronze layer.

        Returns:
            Cleaned and feature-enriched DataFrame ready for the Silver layer.
        """
        data = data.copy()

        # Parse datetimes
        data["tpep_pickup_datetime"] = pd.to_datetime(
            data["tpep_pickup_datetime"]
        )
        data["tpep_dropoff_datetime"] = pd.to_datetime(
            data["tpep_dropoff_datetime"]
        )

        # Handle nulls with configured strategy
        data = self.handle_nulls(data)

        # Remove trips with non-positive passenger count or trip distance
        data = data[
            (data["passenger_count"] > 0) & (data["trip_distance"] > 0)
        ]

        # Remove fare outliers using IQR
        q1 = data["fare_amount"].quantile(0.25)
        q3 = data["fare_amount"].quantile(0.75)
        iqr = q3 - q1
        data = data[
            (data["fare_amount"] >= q1 - 1.5 * iqr)
            & (data["fare_amount"] <= q3 + 1.5 * iqr)
        ]

        # Feature engineering
        data["hour"] = data["tpep_pickup_datetime"].dt.hour
        data["day_of_week"] = data["tpep_pickup_datetime"].dt.dayofweek
        data["month"] = data["tpep_pickup_datetime"].dt.month
        data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

        return data

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Execute the full Bronze -> Silver cleaning pipeline.

        Returns:
            Cleaned DataFrame that has been saved to the Silver layer.
        """
        print(f"Loading bronze data from: {self.bronze_path}")
        raw = self.load_bronze()
        print(f"  Raw rows: {len(raw):,}")

        cleaned = self.clean(raw)
        print(f"  Cleaned rows: {len(cleaned):,}")

        self.save_silver(cleaned)
        return cleaned
