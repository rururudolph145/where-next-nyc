"""Feature engineering: Silver → Gold layer transformation.

Transforms per-trip Silver layer records into an hourly demand panel
with temporal features, cyclical encodings, lag features, and zone
coordinates — ready for model training.
"""

from __future__ import annotations

import os
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from src.shared.config import load_config


class FeatureEngineer:
    """Builds the Gold layer feature set from Silver layer trip data."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialise FeatureEngineer with configuration.

        Args:
            config: Parsed configuration dictionary. If ``None``, loads
                from ``config.yml`` via :func:`src.shared.config.load_config`.
        """
        self.config = config or load_config()
        self.lag_hours: list[int] = self.config["feature"]["lag_hours"]
        self.zone_shapefile_path: str = self.config["feature"]["zone_shapefile_path"]
        self.silver_path: str = self.config["data"]["silver_path"]
        self.gold_path: str = self.config["data"]["gold_path"]

    # ------------------------------------------------------------------
    # Hourly demand aggregation
    # ------------------------------------------------------------------

    def build_hourly_demand(
        self,
        df: pd.DataFrame,
        zone_col: str = "pulocationid",
        datetime_col: str = "tpep_pickup_datetime",
    ) -> pd.DataFrame:
        """Aggregate per-trip records into a ``(zone, hour) → trip_count`` panel.

        Creates a complete grid of all zones × all hours in the date range,
        filling missing combinations with 0.

        Args:
            df: Silver layer DataFrame with at least zone and datetime columns.
            zone_col: Column name for the taxi zone ID.
            datetime_col: Column name for the pickup datetime.

        Returns:
            DataFrame with columns ``[pulocationid, hour, trip_count]``.
        """
        df = df.copy()
        df["hour"] = df[datetime_col].dt.floor("h")

        trip_counts = (
            df.groupby([zone_col, "hour"]).size().rename("trip_count").reset_index()
        )

        all_zones = sorted(trip_counts[zone_col].unique())
        all_hours = pd.date_range(
            start=trip_counts["hour"].min(),
            end=trip_counts["hour"].max(),
            freq="h",
        )
        full_grid = pd.MultiIndex.from_product(
            [all_zones, all_hours], names=[zone_col, "hour"]
        ).to_frame(index=False)

        result = full_grid.merge(trip_counts, on=[zone_col, "hour"], how="left")
        result["trip_count"] = result["trip_count"].fillna(0).astype(int)

        return result

    # ------------------------------------------------------------------
    # Temporal features
    # ------------------------------------------------------------------

    def add_time_features(
        self, df: pd.DataFrame, hour_col: str = "hour"
    ) -> pd.DataFrame:
        """Add temporal features derived from the hour column.

        Args:
            df: DataFrame with an hourly datetime column.
            hour_col: Name of the datetime column.

        Returns:
            DataFrame with new columns: ``hour_of_day``, ``day_of_week``,
            ``month``.
        """
        df = df.copy()
        df["hour_of_day"] = df[hour_col].dt.hour
        df["day_of_week"] = df[hour_col].dt.dayofweek  # 0=Mon, 6=Sun
        df["month"] = df[hour_col].dt.month
        return df

    def add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sine/cosine encoding for cyclical time features.

        Encodes ``hour_of_day`` (period 24), ``day_of_week`` (period 7),
        and ``month`` (period 12) as sin/cos pairs so the model
        understands that e.g. hour 23 and hour 0 are adjacent.

        Args:
            df: DataFrame with ``hour_of_day``, ``day_of_week``, and
                ``month`` columns.

        Returns:
            DataFrame with 6 new columns: ``*_sin`` and ``*_cos`` for
            each feature.
        """
        df = df.copy()
        cyclical: dict[str, int] = {
            "hour_of_day": 24,
            "day_of_week": 7,
            "month": 12,
        }
        for col, period in cyclical.items():
            df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
            df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
        return df

    # ------------------------------------------------------------------
    # Lag features
    # ------------------------------------------------------------------

    def add_lag_features(
        self,
        df: pd.DataFrame,
        zone_col: str = "pulocationid",
        target_col: str = "trip_count",
    ) -> pd.DataFrame:
        """Add lagged pickup count columns per zone.

        For each value in ``self.lag_hours``, creates a column
        ``trip_count_lag_{h}`` containing the pickup count ``h`` hours
        earlier for the same zone.

        The DataFrame must be sorted by ``(zone, hour)`` and contain a
        complete hourly grid so that a simple ``shift(h)`` within each
        zone group produces the correct lag.

        Args:
            df: DataFrame with zone ID, hour, and trip_count columns.
            zone_col: Column name for the zone ID.
            target_col: Column name for the pickup count.

        Returns:
            DataFrame with new lag columns. Rows where any lag is NaN
            (i.e. the first ``max(lag_hours)`` rows per zone) are dropped.
        """
        df = df.sort_values([zone_col, "hour"]).copy()
        for h in self.lag_hours:
            df[f"{target_col}_lag_{h}"] = df.groupby(zone_col)[target_col].shift(h)

        lag_cols = [f"{target_col}_lag_{h}" for h in self.lag_hours]
        df = df.dropna(subset=lag_cols)
        for h in self.lag_hours:
            df[f"{target_col}_lag_{h}"] = df[f"{target_col}_lag_{h}"].astype(int)

        return df

    # ------------------------------------------------------------------
    # Zone coordinates
    # ------------------------------------------------------------------

    def add_zone_coordinates(
        self,
        df: pd.DataFrame,
        zone_col: str = "pulocationid",
    ) -> pd.DataFrame:
        """Merge zone centroid coordinates onto the DataFrame.

        Each zone is represented by the centroid of its polygon,
        reprojected to WGS84.

        Args:
            df: DataFrame containing a ``pulocationid`` column.
            zone_col: Column name for the zone ID.

        Returns:
            DataFrame with two new columns: ``pickup_longitude``,
            ``pickup_latitude``.
        """
        gdf = gpd.read_file(self.zone_shapefile_path).to_crs("EPSG:4326")
        centroids = gdf.to_crs("EPSG:3857").geometry.centroid.to_crs("EPSG:4326")
        lookup = pd.DataFrame(
            {
                "LocationID": gdf["LocationID"].astype(int),
                "longitude": centroids.x,
                "latitude": centroids.y,
            }
        ).drop_duplicates(subset="LocationID")

        lon_map = lookup.set_index("LocationID")["longitude"]
        lat_map = lookup.set_index("LocationID")["latitude"]

        df = df.copy()
        df["pickup_longitude"] = df[zone_col].map(lon_map)
        df["pickup_latitude"] = df[zone_col].map(lat_map)

        # Drop zones with no shapefile match (e.g. 57, 264, 265)
        n_before = len(df)
        df = df.dropna(subset=["pickup_longitude", "pickup_latitude"])
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"  Dropped {n_dropped:,} rows with unknown zone coordinates")

        return df

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self, data: pd.DataFrame | None = None) -> pd.DataFrame:
        """Execute the full Silver → Gold feature engineering pipeline.

        Args:
            data: Optional Silver layer DataFrame. If ``None``, reads from
                the configured ``data.silver_path``.

        Returns:
            Gold layer DataFrame.
        """
        if data is None:
            data = pd.read_parquet(self.silver_path)

        panel = self.build_hourly_demand(data)
        panel = self.add_time_features(panel)
        panel = self.add_cyclical_features(panel)
        panel = self.add_lag_features(panel)
        panel = self.add_zone_coordinates(panel)

        os.makedirs(self.gold_path, exist_ok=True)
        output_path = os.path.join(self.gold_path, "hourly_demand_featured.parquet")
        panel.to_parquet(output_path, index=False)
        print(f"Saved → {output_path}  ({len(panel):,} rows)")

        return panel
