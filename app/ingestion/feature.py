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
from sklearn.neighbors import BallTree  # type: ignore[import-untyped]

from app.core.config import load_config


def build_centroid_lookup(shapefile_path: str) -> dict[int, tuple[float, float]]:
    """Return ``{LocationID: (lat_3dp, lon_3dp)}`` from shapefile centroids.

    Reads the taxi zones shapefile, projects geometries to EPSG:3857 to
    compute Euclidean centroids, then reprojects to WGS84 (EPSG:4326) to
    obtain geographic coordinates. Values are rounded to 3 decimal places
    at build time. Zones with null geometry are absent from the returned dict.

    Args:
        shapefile_path: Filesystem path to the taxi zones ``.shp`` file.

    Returns:
        Dictionary mapping integer ``LocationID`` to a ``(latitude, longitude)``
        tuple, both rounded to 3 decimal places.
    """
    gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    centroids = gdf.to_crs("EPSG:3857").geometry.centroid.to_crs("EPSG:4326")
    result: dict[int, tuple[float, float]] = {}
    for loc_id, lon, lat in zip(
        gdf["LocationID"].astype(int), centroids.x, centroids.y, strict=True
    ):
        if pd.isna(lon) or pd.isna(lat):
            continue
        result[int(loc_id)] = (round(float(lat), 3), round(float(lon), 3))
    return result


class FeatureEngineer:
    """Builds the Gold layer feature set from Silver layer trip data."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialise FeatureEngineer with configuration.

        Args:
            config: Parsed configuration dictionary. If ``None``, loads
                from ``config.yml`` via :func:`app.core.config.load_config`.
        """
        self.config = config or load_config()
        # Aggregation bucket width (pandas offset alias). Defaults to "h"
        # so existing configs without the key keep their hourly behavior.
        # PBI A1 introduces this knob; PBI A2/A3 build on top of it for
        # 15-min refresh and shifted-horizon forecasting.
        self.bucket: str = self.config["feature"].get("bucket", "h")
        # ``feature.lags`` is the canonical key (values are bucket offsets,
        # not hours). PBI A2 generalizes the unit; the legacy
        # ``feature.lag_hours`` key is accepted as a back-compat alias so
        # existing configs and notebooks keep working. When both are
        # present, ``lags`` wins. ``self.lag_hours`` mirrors ``self.lags``
        # so downstream code that still reads the old attribute name
        # continues to work.
        feature_cfg = self.config["feature"]
        if "lags" in feature_cfg:
            self.lags: list[int] = feature_cfg["lags"]
        else:
            self.lags = feature_cfg["lag_hours"]
        self.lag_hours: list[int] = self.lags
        # Forecast horizon expressed in *buckets*. Default 0 = predict the
        # current bucket (legacy ``trip_count``). Set to 2 under bucket="15min"
        # for the t+30 design (see docs/design_t30_15min_refresh.html §4.2).
        # PBI A3 introduces the knob; downstream consumers (trainer, simulator,
        # heatmap API) point at the resulting ``target_t{N}`` column.
        self.horizon_buckets: int = feature_cfg.get("horizon_buckets", 0)
        # PBI B2: number of *future* buckets to sum into a single target column.
        # Under bucket="15min" with horizon_sum_buckets=2 this builds the
        # 30-min lookahead target = shift(-1) + shift(-2) of trip_count, i.e.
        # the "extended nowcast" design. 0 disables the feature so B1 stays
        # the default once horizon_sum_buckets is omitted from config.
        self.horizon_sum_buckets: int = feature_cfg.get("horizon_sum_buckets", 0)
        # Number of nearest neighbours used by ``add_neighbor_lag_smoothing``
        # (PBI B1). Default 0 disables the smoothing path so legacy configs
        # without the key keep their pre-B1 Gold schema.
        self.n_neighbors: int = feature_cfg.get("n_neighbors", 0)
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
        df["hour"] = df[datetime_col].dt.floor(self.bucket)

        trip_counts = (
            df.groupby([zone_col, "hour"]).size().rename("trip_count").reset_index()
        )

        all_zones = sorted(trip_counts[zone_col].unique())
        all_hours = pd.date_range(
            start=trip_counts["hour"].min(),
            end=trip_counts["hour"].max(),
            freq=self.bucket,
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

        For each value ``n`` in ``self.lags``, creates a column
        ``trip_count_lag_{n}`` containing the pickup count ``n`` *buckets*
        earlier for the same zone. Under the default ``bucket="h"``, one
        bucket equals one hour (legacy semantics). Under ``bucket="15min"``,
        ``lag_1`` is 15 minutes earlier and so on.

        The DataFrame must be sorted by ``(zone, hour)`` and contain a
        complete bucket grid so that a simple ``shift(n)`` within each
        zone group produces the correct lag.

        Args:
            df: DataFrame with zone ID, hour, and trip_count columns.
            zone_col: Column name for the zone ID.
            target_col: Column name for the pickup count.

        Returns:
            DataFrame with new lag columns. Rows where any lag is NaN
            (i.e. the first ``max(lags)`` rows per zone) are dropped.
        """
        df = df.sort_values([zone_col, "hour"]).copy()
        for n in self.lags:
            df[f"{target_col}_lag_{n}"] = df.groupby(zone_col)[target_col].shift(n)

        lag_cols = [f"{target_col}_lag_{n}" for n in self.lags]
        df = df.dropna(subset=lag_cols)
        for n in self.lags:
            df[f"{target_col}_lag_{n}"] = df[f"{target_col}_lag_{n}"].astype(int)

        return df

    # ------------------------------------------------------------------
    # Shifted target (forecast horizon)
    # ------------------------------------------------------------------

    def add_shifted_target(
        self,
        df: pd.DataFrame,
        zone_col: str = "pulocationid",
        target_col: str = "trip_count",
    ) -> pd.DataFrame:
        """Add a forward-shifted target column ``target_t{horizon_buckets}``.

        For each ``(zone, bucket=t)`` row, the new column contains the
        ``target_col`` value from ``(zone, bucket=t + horizon_buckets*bucket_width)``
        — the demand we want to *predict* from features observable at ``t``.

        Implementation uses ``groupby(zone).shift(-horizon_buckets)``. The
        negative sign points the shift FORWARD in time (target = future
        demand). A positive sign would produce the opposite — past as target —
        which is the classic silent leakage bug. PBI A3 locks the sign with
        ``tests/test_feature.py::TestShiftedTarget``.

        When ``self.horizon_buckets == 0`` the target equals the current
        bucket's ``target_col`` (legacy behavior) and no rows are dropped.
        For positive horizons, the trailing ``horizon_buckets`` rows per
        zone become NaN (no future to observe) and are dropped.

        Args:
            df: DataFrame sorted by ``(zone, hour)`` with a complete bucket
                grid.
            zone_col: Column name for the zone ID.
            target_col: Column name for the demand to be shifted.

        Returns:
            DataFrame with a new ``target_t{horizon_buckets}`` column.
            Trailing NaN rows (per zone) are dropped when horizon > 0.
        """
        n = self.horizon_buckets
        out_col = f"target_t{n}"
        df = df.sort_values([zone_col, "hour"]).copy()
        if n == 0:
            df[out_col] = df[target_col]
            return df
        df[out_col] = df.groupby(zone_col)[target_col].shift(-n)
        df = df.dropna(subset=[out_col])
        df[out_col] = df[out_col].astype(int)
        return df

    # ------------------------------------------------------------------
    # Summed horizon target (PBI B2)
    # ------------------------------------------------------------------

    def add_summed_horizon_target(
        self,
        df: pd.DataFrame,
        zone_col: str = "pulocationid",
        target_col: str = "trip_count",
    ) -> pd.DataFrame:
        """Add a forward-summed target column ``target_sum_t1_t{n}``.

        Computes::

            target_sum_t1_tN(zone, t) = sum_{k=1..N} target_col(zone, t + k*bucket)

        This is the "extended nowcast" target used by PBI B2: under
        ``bucket="15min"`` with ``horizon_sum_buckets=2`` the column equals
        ``demand[t+15..t+30) + demand[t+30..t+45)`` — the cumulative pickups
        over the next 30 minutes. (See `docs/design_t30_15min_refresh.html`
        §4.2 for why a sum is the right shape for driver-facing demand.)

        Distinct from :meth:`add_shifted_target` (PBI A3) which emits a
        *single* future bucket. B2 emits a *sum across N future buckets*.

        Implementation uses ``groupby(zone).shift(-k)`` for k=1..N. The
        negative sign points the shift FORWARD in time (target = future
        demand). Positive would produce past-as-target — the classic
        silent leakage bug — and is locked out by
        ``tests/test_feature.py::TestSummedHorizonTarget``.

        Args:
            df: DataFrame sorted by ``(zone, hour)`` with a complete bucket
                grid.
            zone_col: Column name for the zone ID.
            target_col: Column name for the demand to be summed.

        Returns:
            DataFrame with a new ``target_sum_t1_t{N}`` column. The
            trailing ``N`` rows per zone (where any contributing shift is
            NaN) are dropped, so the output is shorter than the input.
        """
        n = self.horizon_sum_buckets
        if n <= 0:
            return df
        out_col = f"target_sum_t1_t{n}"
        df = df.sort_values([zone_col, "hour"]).copy()

        # Build each future-shift column, then sum. We materialise
        # intermediates rather than chaining .add() so the dropna step
        # below can see all NaNs in one pass.
        shift_cols = []
        for k in range(1, n + 1):
            col = f"_shift_{k}"
            df[col] = df.groupby(zone_col)[target_col].shift(-k)
            shift_cols.append(col)

        df[out_col] = df[shift_cols].sum(axis=1, skipna=False)
        df = df.drop(columns=shift_cols)
        df = df.dropna(subset=[out_col])
        df[out_col] = df[out_col].astype(int)
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
        reprojected to WGS84. Delegates centroid computation to
        :func:`build_centroid_lookup`.

        Args:
            df: DataFrame containing a ``pulocationid`` column.
            zone_col: Column name for the zone ID.

        Returns:
            DataFrame with two new columns: ``pickup_longitude``,
            ``pickup_latitude``.
        """
        centroid_lookup = build_centroid_lookup(self.zone_shapefile_path)
        lon_map = {loc_id: coords[1] for loc_id, coords in centroid_lookup.items()}
        lat_map = {loc_id: coords[0] for loc_id, coords in centroid_lookup.items()}

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
    # Spatial smoothing (PBI B1)
    # ------------------------------------------------------------------

    def add_neighbor_lag_smoothing(
        self,
        df: pd.DataFrame,
        zone_col: str = "pulocationid",
        lag_col: str = "trip_count_lag_1",
    ) -> pd.DataFrame:
        """Add neighbour-mean smoothing of the lag_1 column.

        For each zone we find its ``self.n_neighbors`` nearest peers by
        great-circle distance over the (longitude, latitude) centroids
        already attached by :meth:`add_zone_coordinates`. For every
        ``(zone, bucket)`` row we then write the mean of those neighbours'
        ``lag_col`` values at the *same* bucket into a new column named
        ``{lag_col}_neighbor_mean``.

        Design choices locked by ``tests/test_feature.py::TestNeighborLagSmoothing``:

        * **Mean, not sum** — keeps the smoothed feature on the same count
          scale as ``lag_col`` so a single coefficient covers both.
        * **Self excluded** — only the k nearest *other* zones contribute,
          so a zone with no nearby demand can't smuggle its own value back
          in via the smoother.
        * **Reads lag_1, not trip_count** — the smoother feeds on lagged
          history, never on the current bucket's count. This is the
          anti-leakage invariant guarded by
          ``tests/test_e2e_synthetic.py::test_no_target_leakage``.

        Args:
            df: Gold-layer panel with ``zone_col``, ``hour``,
                ``pickup_longitude``, ``pickup_latitude``, and ``lag_col``.
            zone_col: Zone-id column.
            lag_col: Source lag column to smooth (default ``trip_count_lag_1``).

        Returns:
            DataFrame with new ``{lag_col}_neighbor_mean`` column.
        """
        out_col = f"{lag_col}_neighbor_mean"
        k = self.n_neighbors
        if k <= 0:
            # Smoothing disabled — leave the panel untouched.
            return df

        # One representative centroid per zone (haversine wants radians).
        # Drop zones with NaN coordinates — BallTree rejects them. Their
        # rows fall out of the merge below as NaN, signalling the upstream
        # bug rather than silently fabricating a neighbour-mean.
        zone_centroids = (
            df[[zone_col, "pickup_longitude", "pickup_latitude"]]
            .dropna(subset=["pickup_longitude", "pickup_latitude"])
            .drop_duplicates(subset=[zone_col])
            .reset_index(drop=True)
        )
        coords_rad = np.radians(
            zone_centroids[["pickup_latitude", "pickup_longitude"]].to_numpy()
        )
        n_zones = len(zone_centroids)
        # Cap k at n_zones-1 so we always have *some* neighbour to mean over
        # (tiny test fixtures with 3-4 zones would otherwise blow up).
        effective_k = min(k, n_zones - 1)
        if effective_k <= 0:
            df = df.copy()
            df[out_col] = np.nan
            return df

        tree = BallTree(coords_rad, metric="haversine")
        # +1 because the closest match is always the zone itself; we drop it.
        _, idx = tree.query(coords_rad, k=effective_k + 1)
        neighbour_idx = idx[:, 1:]  # shape: (n_zones, effective_k)

        zone_ids = zone_centroids[zone_col].to_numpy()
        # neighbours_by_zone[z] = list of neighbour zone-ids for zone z
        neighbours_by_zone: dict[Any, list[Any]] = {
            zone_ids[i]: zone_ids[neighbour_idx[i]].tolist() for i in range(n_zones)
        }

        # Pivot lag_col to a (hour × zone) matrix so neighbour-mean is just
        # a per-row column selection + mean. Much cheaper than a per-row loop.
        wide = df.pivot_table(
            index="hour", columns=zone_col, values=lag_col, aggfunc="first"
        )
        neighbour_means = pd.DataFrame(index=wide.index, columns=wide.columns)
        for z in wide.columns:
            cols = [n for n in neighbours_by_zone.get(z, []) if n in wide.columns]
            if not cols:
                neighbour_means[z] = np.nan
            else:
                neighbour_means[z] = wide[cols].mean(axis=1)

        # Name the columns axis so stack().reset_index() produces a tidy
        # (hour, zone_col, value) frame without a fragile "level_1" label.
        neighbour_means.columns.name = zone_col
        stacked: pd.Series[Any] = neighbour_means.stack()  # type: ignore[assignment]
        long = stacked.rename(out_col).reset_index()
        return df.merge(long, on=["hour", zone_col], how="left")

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

        # Normalise column names to lowercase for consistent downstream use
        data.columns = pd.Index([c.lower() for c in data.columns])

        panel = self.build_hourly_demand(data)
        panel = self.add_time_features(panel)
        panel = self.add_cyclical_features(panel)
        panel = self.add_lag_features(panel)
        # Only materialize a shifted-target column when explicitly configured.
        # Under the default ``horizon_buckets=0`` the trainer still reads the
        # legacy ``trip_count`` column, so Gold's schema stays bit-identical
        # for back-compat with existing training/serving paths.
        if self.horizon_buckets > 0:
            panel = self.add_shifted_target(panel)
        # PBI B2: extended-nowcast target = sum of next N future buckets.
        # Independent of horizon_buckets — typical config uses only one or
        # the other. When both are set, both columns are emitted and the
        # trainer picks which one via ``train.target``.
        if self.horizon_sum_buckets > 0:
            panel = self.add_summed_horizon_target(panel)
        panel = self.add_zone_coordinates(panel)
        # Spatial smoothing must run *after* zone coordinates (needs lat/lon)
        # and *after* the orphan-zone drop, so the neighbour set is stable.
        # No-op when ``feature.n_neighbors`` is unset/0 (legacy back-compat).
        if self.n_neighbors > 0:
            panel = self.add_neighbor_lag_smoothing(panel)

        os.makedirs(self.gold_path, exist_ok=True)
        output_path = os.path.join(self.gold_path, "hourly_demand_featured.parquet")
        panel.to_parquet(output_path, index=False)
        print(f"Saved → {output_path}  ({len(panel):,} rows)")

        return panel
