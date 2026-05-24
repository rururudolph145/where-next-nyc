"""Tests for the feature engineering module (app/ingestion/feature.py)."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from app.ingestion.feature import FeatureEngineer, build_centroid_lookup

# -- Shared config fixture used across tests --------------------------------

MINIMAL_CONFIG: dict[str, Any] = {
    "env": "local",
    "data": {
        "silver_path": "data/silver/test.parquet",
        "gold_path": "data/gold/yellow_tripdata",
    },
    "feature": {
        "lag_hours": [1, 3],
        "zone_shapefile_path": "data/reference/taxi_zones.shp",
    },
}


@pytest.fixture
def fe() -> FeatureEngineer:
    """Return a FeatureEngineer with a minimal test config."""
    return FeatureEngineer(config=MINIMAL_CONFIG)


# -- Helpers to build sample data -------------------------------------------


def _make_silver_df() -> pd.DataFrame:
    """Create a small Silver-like DataFrame with 2 zones and 4 hours."""
    rows: list[dict[str, object]] = []
    for zone in [1, 2]:
        for hour_offset in range(4):
            ts = pd.Timestamp("2022-01-01") + pd.Timedelta(hours=hour_offset)
            # Zone 1: 2 trips per hour; Zone 2: 1 trip per hour
            n_trips = 2 if zone == 1 else 1
            for _ in range(n_trips):
                rows.append(
                    {
                        "tpep_pickup_datetime": ts + pd.Timedelta(minutes=10),
                        "pulocationid": zone,
                    }
                )
    return pd.DataFrame(rows)


def _make_hourly_panel() -> pd.DataFrame:
    """Create an hourly demand panel for testing time/cyclical/lag features."""
    zones = [1, 2]
    hours = pd.date_range("2022-01-01", periods=48, freq="h")
    records: list[dict[str, object]] = []
    for zone in zones:
        for i, hour in enumerate(hours):
            records.append(
                {
                    "pulocationid": zone,
                    "hour": hour,
                    "trip_count": (i + 1) * zone,
                }
            )
    return pd.DataFrame(records)


# -- Test: build_hourly_demand ---------------------------------------------


class TestBuildHourlyDemand:
    """Tests for FeatureEngineer.build_hourly_demand."""

    def test_complete_grid(self, fe: FeatureEngineer) -> None:
        """Output must have n_zones × n_hours rows (complete grid)."""
        silver = _make_silver_df()
        result = fe.build_hourly_demand(silver)

        n_zones = silver["pulocationid"].nunique()
        n_hours = len(
            pd.date_range(
                start=silver["tpep_pickup_datetime"].dt.floor("h").min(),
                end=silver["tpep_pickup_datetime"].dt.floor("h").max(),
                freq="h",
            )
        )
        assert len(result) == n_zones * n_hours

    def test_correct_counts(self, fe: FeatureEngineer) -> None:
        """Trip counts match the number of trips per zone-hour."""
        silver = _make_silver_df()
        result = fe.build_hourly_demand(silver)

        zone1 = result[result["pulocationid"] == 1]
        zone2 = result[result["pulocationid"] == 2]
        assert (zone1["trip_count"] == 2).all()
        assert (zone2["trip_count"] == 1).all()

    def test_output_columns(self, fe: FeatureEngineer) -> None:
        """Output has the expected columns."""
        silver = _make_silver_df()
        result = fe.build_hourly_demand(silver)
        assert list(result.columns) == [
            "pulocationid",
            "hour",
            "trip_count",
        ]

    def test_zero_fill(self, fe: FeatureEngineer) -> None:
        """Zone-hours with no trips are filled with 0."""
        silver = pd.DataFrame(
            {
                "tpep_pickup_datetime": pd.to_datetime(
                    [
                        "2022-01-01 00:10:00",
                        "2022-01-01 02:10:00",
                    ]
                ),
                "pulocationid": [1, 1],
            }
        )
        result = fe.build_hourly_demand(silver)
        hour_1 = result[result["hour"] == pd.Timestamp("2022-01-01 01:00:00")]
        assert (hour_1["trip_count"] == 0).all()


# -- PBI A1: Configurable bucket width --------------------------------------
#
# Locks the contract that the aggregation bucket is a config knob, not a
# hard-coded "h". Default ("h") must reproduce legacy behavior bit-identically;
# "15min" must produce 4× the rows over the same time range.


class TestBucketParameterization:
    """Tests for the ``feature.bucket`` configuration knob (PBI A1)."""

    @staticmethod
    def _config(bucket: str | None) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            **MINIMAL_CONFIG,
            "feature": {**MINIMAL_CONFIG["feature"]},
        }
        if bucket is not None:
            cfg["feature"]["bucket"] = bucket
        return cfg

    def test_default_bucket_matches_hourly_legacy(self) -> None:
        """Omitting ``feature.bucket`` must reproduce hourly-legacy output."""
        silver = _make_silver_df()

        legacy = FeatureEngineer(config=self._config(bucket=None)).build_hourly_demand(
            silver
        )
        explicit = FeatureEngineer(config=self._config(bucket="h")).build_hourly_demand(
            silver
        )

        pd.testing.assert_frame_equal(
            legacy.reset_index(drop=True), explicit.reset_index(drop=True)
        )

    def test_15min_bucket_yields_finer_grid(self) -> None:
        """``bucket='15min'`` must produce a 15-min grid that strictly contains
        every hourly bucket and adds 3 sub-buckets between them.

        Inclusive ``pd.date_range`` means an hourly grid of ``N`` rows covers
        ``N-1`` hours, while the matching 15-min grid covers the same span
        with ``4*(N-1) + 1`` rows. The test pins this exact relationship so
        regressions to floor("h") would fail immediately.
        """
        silver = _make_silver_df()

        hourly = FeatureEngineer(config=self._config(bucket="h")).build_hourly_demand(
            silver
        )
        quarter = FeatureEngineer(
            config=self._config(bucket="15min")
        ).build_hourly_demand(silver)

        n_zones = silver["pulocationid"].nunique()
        n_hours_per_zone = len(hourly) // n_zones
        expected_quarters_per_zone = 4 * (n_hours_per_zone - 1) + 1
        assert len(quarter) == n_zones * expected_quarters_per_zone

        # Every hourly timestamp must appear in the 15-min grid.
        assert set(hourly["hour"]).issubset(set(quarter["hour"]))

    def test_15min_bucket_preserves_trip_totals(self) -> None:
        """Switching to 15-min buckets must not change total trips per zone."""
        silver = _make_silver_df()

        hourly = FeatureEngineer(config=self._config(bucket="h")).build_hourly_demand(
            silver
        )
        quarter = FeatureEngineer(
            config=self._config(bucket="15min")
        ).build_hourly_demand(silver)

        hourly_totals = hourly.groupby("pulocationid")["trip_count"].sum()
        quarter_totals = quarter.groupby("pulocationid")["trip_count"].sum()
        pd.testing.assert_series_equal(hourly_totals, quarter_totals)


# -- PBI A2: Generalize lags to bucket units --------------------------------
#
# Pins the contract that ``feature.lags`` is the canonical key (values are
# interpreted as bucket offsets, not hours). The legacy ``feature.lag_hours``
# key remains accepted as a back-compat alias so existing configs and notebooks
# continue to work. When both keys are present, ``lags`` wins.


class TestLagsConfig:
    """Tests for the ``feature.lags`` configuration knob (PBI A2)."""

    @staticmethod
    def _config(
        lags: list[int] | None = None, lag_hours: list[int] | None = None
    ) -> dict[str, Any]:
        feature: dict[str, Any] = {
            "zone_shapefile_path": MINIMAL_CONFIG["feature"]["zone_shapefile_path"],
        }
        if lags is not None:
            feature["lags"] = lags
        if lag_hours is not None:
            feature["lag_hours"] = lag_hours
        return {**MINIMAL_CONFIG, "feature": feature}

    def test_lags_key_is_honored(self) -> None:
        """``feature.lags`` must produce lag columns matching the configured offsets."""
        fe = FeatureEngineer(config=self._config(lags=[1, 2]))
        panel = _make_hourly_panel()
        result = fe.add_lag_features(panel)

        assert "trip_count_lag_1" in result.columns
        assert "trip_count_lag_2" in result.columns
        # Spot-check: within each zone, ``lag_1`` equals ``trip_count`` from the
        # previous bucket. ``add_lag_features`` drops rows where any lag is NaN,
        # so after lags=[1, 2] the first 2 rows per zone are gone; the reference
        # is the original ``trip_count`` shifted by 1 with the matching rows
        # taken from the result's index slice.
        for zone in result["pulocationid"].unique():
            zone_actual = result[result["pulocationid"] == zone].sort_values("hour")
            zone_panel = panel[panel["pulocationid"] == zone].sort_values("hour")
            reference = (
                zone_panel["trip_count"]
                .shift(1)
                .iloc[2:]
                .astype(int)
                .reset_index(drop=True)
            )
            actual = zone_actual["trip_count_lag_1"].reset_index(drop=True)
            pd.testing.assert_series_equal(actual, reference, check_names=False)

    def test_legacy_lag_hours_back_compat(self) -> None:
        """Configs using only ``feature.lag_hours`` (no ``lags``) must keep working."""
        fe = FeatureEngineer(config=self._config(lag_hours=[1, 3]))
        panel = _make_hourly_panel()
        result = fe.add_lag_features(panel)
        for h in [1, 3]:
            assert f"trip_count_lag_{h}" in result.columns

    def test_lags_takes_precedence_over_lag_hours(self) -> None:
        """When both keys are present, ``lags`` wins."""
        fe = FeatureEngineer(config=self._config(lags=[1, 2], lag_hours=[5, 7]))
        assert fe.lags == [1, 2]
        # Old alias mirrors the canonical attribute, not the deprecated key.
        assert fe.lag_hours == [1, 2]

    def test_missing_both_keys_raises(self) -> None:
        """Omitting both ``lags`` and ``lag_hours`` must raise a KeyError."""
        with pytest.raises(KeyError):
            FeatureEngineer(config=self._config(lags=None, lag_hours=None))


# -- PBI A3: Shifted-target generation --------------------------------------
#
# Pins the contract that ``feature.horizon_buckets`` produces a ``target_t{N}``
# column via ``groupby(zone).shift(-N)``. Direction of shift is critical:
# ``shift(-N)`` looks *forward* (target = future demand), whereas
# ``shift(+N)`` would leak past as target — the classic silent-leakage bug.
# These tests lock the contract before any consumer (trainer, simulator,
# heatmap API) depends on it.


class TestShiftedTarget:
    """Tests for ``FeatureEngineer.add_shifted_target`` (PBI A3)."""

    @staticmethod
    def _config(horizon_buckets: int | None) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            **MINIMAL_CONFIG,
            "feature": {**MINIMAL_CONFIG["feature"]},
        }
        if horizon_buckets is not None:
            cfg["feature"]["horizon_buckets"] = horizon_buckets
        return cfg

    def test_horizon_zero_equals_trip_count(self) -> None:
        """``horizon_buckets=0`` → ``target_t0`` equals ``trip_count`` row-for-row.

        No rows are dropped because no shift produces NaN.
        """
        fe = FeatureEngineer(config=self._config(horizon_buckets=0))
        panel = _make_hourly_panel()
        result = fe.add_shifted_target(panel)

        assert "target_t0" in result.columns
        assert len(result) == len(panel)
        pd.testing.assert_series_equal(
            result["target_t0"].astype(int).reset_index(drop=True),
            result["trip_count"].astype(int).reset_index(drop=True),
            check_names=False,
        )

    def test_horizon_two_shifts_forward(self) -> None:
        """``horizon_buckets=2`` → for each ``(zone, t)``, ``target_t2`` equals
        ``trip_count`` at ``(zone, t + 2*bucket)``.

        Trailing 2 rows per zone are dropped (no future to look at).
        """
        fe = FeatureEngineer(config=self._config(horizon_buckets=2))
        panel = _make_hourly_panel()
        n_zones = panel["pulocationid"].nunique()
        n_rows_per_zone = len(panel) // n_zones

        result = fe.add_shifted_target(panel)

        assert "target_t2" in result.columns
        # 2 trailing rows dropped per zone
        assert len(result) == n_zones * (n_rows_per_zone - 2)

        # For each zone, target_t2 at bucket t equals trip_count at bucket t+2
        for zone in result["pulocationid"].unique():
            panel_zone = (
                panel[panel["pulocationid"] == zone]
                .sort_values("hour")
                .reset_index(drop=True)
            )
            result_zone = (
                result[result["pulocationid"] == zone]
                .sort_values("hour")
                .reset_index(drop=True)
            )
            expected = panel_zone["trip_count"].shift(-2).dropna().astype(int)
            actual = result_zone["target_t2"].astype(int).reset_index(drop=True)
            pd.testing.assert_series_equal(
                actual, expected.reset_index(drop=True), check_names=False
            )

    def test_shift_direction_is_forward_not_backward(self) -> None:
        """Regression guard: ``target_t{N}`` must point at the FUTURE, not the past.

        If a refactor ever flips the sign in ``shift(-N)`` → ``shift(+N)``, the
        model would train on past observations as the "target", producing
        leakage-perfect backtest scores that collapse in production. This test
        constructs a panel with monotonically-increasing trip_count so a forward
        shift yields strictly larger targets than the row's own trip_count.
        """
        # Strictly increasing trip_count over time within a single zone.
        panel = pd.DataFrame(
            {
                "pulocationid": [1] * 10,
                "hour": pd.date_range("2022-01-01", periods=10, freq="h"),
                "trip_count": list(range(10)),
            }
        )
        fe = FeatureEngineer(config=self._config(horizon_buckets=2))
        result = fe.add_shifted_target(panel)

        # Every retained row's target must be strictly greater than its
        # own trip_count (because trip_count is strictly increasing and
        # we shifted forward by 2).
        assert (result["target_t2"] > result["trip_count"]).all()

    def test_no_cross_zone_bleed(self) -> None:
        """Shifting in zone A must not pull rows from zone B's tail in as targets."""
        # Two zones; zone 1 has small values, zone 2 has large values.
        # If shift leaked across zones, zone 1's tail would carry zone 2's first
        # values (or zone 2's tail would be NaN-but-not-dropped).
        panel = pd.DataFrame(
            {
                "pulocationid": [1] * 5 + [2] * 5,
                "hour": (
                    list(pd.date_range("2022-01-01", periods=5, freq="h"))
                    + list(pd.date_range("2022-01-01", periods=5, freq="h"))
                ),
                "trip_count": [1, 2, 3, 4, 5] + [100, 200, 300, 400, 500],
            }
        )
        fe = FeatureEngineer(config=self._config(horizon_buckets=2))
        result = fe.add_shifted_target(panel)

        # 2 trailing rows per zone dropped → 3 retained per zone
        assert len(result) == 6
        zone1 = result[result["pulocationid"] == 1].sort_values("hour")
        # Zone 1's targets must all be small (< 100). If cross-zone bleed
        # occurred, the last retained row of zone 1 would carry zone 2's
        # first value (100) — this assertion catches that.
        assert (zone1["target_t2"] < 100).all()

    def test_default_horizon_is_zero(self) -> None:
        """Omitting ``feature.horizon_buckets`` must default to 0 (legacy)."""
        fe = FeatureEngineer(config=self._config(horizon_buckets=None))
        assert fe.horizon_buckets == 0


# -- Test: add_time_features -----------------------------------------------


class TestAddTimeFeatures:
    """Tests for FeatureEngineer.add_time_features."""

    def test_columns_created(self, fe: FeatureEngineer) -> None:
        """hour_of_day, day_of_week, month columns are added."""
        panel = _make_hourly_panel()
        result = fe.add_time_features(panel)
        for col in ["hour_of_day", "day_of_week", "month"]:
            assert col in result.columns

    def test_correct_values(self, fe: FeatureEngineer) -> None:
        """Extracted values match the datetime components."""
        panel = _make_hourly_panel()
        result = fe.add_time_features(panel)

        expected_hours = result["hour"].dt.hour
        expected_dow = result["hour"].dt.dayofweek
        expected_month = result["hour"].dt.month

        pd.testing.assert_series_equal(
            result["hour_of_day"], expected_hours, check_names=False
        )
        pd.testing.assert_series_equal(
            result["day_of_week"], expected_dow, check_names=False
        )
        pd.testing.assert_series_equal(
            result["month"], expected_month, check_names=False
        )

    def test_preserves_original_columns(self, fe: FeatureEngineer) -> None:
        """Original columns are not dropped."""
        panel = _make_hourly_panel()
        original_cols = set(panel.columns)
        result = fe.add_time_features(panel)
        assert original_cols.issubset(set(result.columns))


# -- Test: add_cyclical_features -------------------------------------------


class TestAddCyclicalFeatures:
    """Tests for FeatureEngineer.add_cyclical_features."""

    def test_columns_created(self, fe: FeatureEngineer) -> None:
        """Six sin/cos columns are added."""
        panel = fe.add_time_features(_make_hourly_panel())
        result = fe.add_cyclical_features(panel)
        expected = [
            "hour_of_day_sin",
            "hour_of_day_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "month_sin",
            "month_cos",
        ]
        for col in expected:
            assert col in result.columns

    def test_value_range(self, fe: FeatureEngineer) -> None:
        """All sin/cos values are in [-1, 1]."""
        panel = fe.add_time_features(_make_hourly_panel())
        result = fe.add_cyclical_features(panel)
        for suffix in ["_sin", "_cos"]:
            for base in ["hour_of_day", "day_of_week", "month"]:
                col = f"{base}{suffix}"
                assert result[col].min() >= -1.0
                assert result[col].max() <= 1.0

    def test_pythagorean_identity(self, fe: FeatureEngineer) -> None:
        """sin²(x) + cos²(x) ≈ 1 for all rows."""
        panel = fe.add_time_features(_make_hourly_panel())
        result = fe.add_cyclical_features(panel)
        for base in ["hour_of_day", "day_of_week", "month"]:
            sin2 = result[f"{base}_sin"] ** 2
            cos2 = result[f"{base}_cos"] ** 2
            np.testing.assert_allclose(sin2 + cos2, 1.0, atol=1e-10)


# -- Test: add_lag_features -------------------------------------------------


class TestAddLagFeatures:
    """Tests for FeatureEngineer.add_lag_features."""

    def test_lag_columns_created(self, fe: FeatureEngineer) -> None:
        """Lag columns are created for each configured lag hour."""
        panel = _make_hourly_panel()
        result = fe.add_lag_features(panel)
        for h in fe.lag_hours:
            assert f"trip_count_lag_{h}" in result.columns

    def test_lag_values_correct(self, fe: FeatureEngineer) -> None:
        """Lag values match trip_count from h hours prior within the same zone."""
        panel = _make_hourly_panel()
        result = fe.add_lag_features(panel)

        zone1 = result[result["pulocationid"] == 1].sort_values("hour")
        for _, row in zone1.iterrows():
            for h in fe.lag_hours:
                lag_hour = row["hour"] - pd.Timedelta(hours=h)
                prior = panel[
                    (panel["pulocationid"] == 1) & (panel["hour"] == lag_hour)
                ]
                if not prior.empty:
                    assert row[f"trip_count_lag_{h}"] == prior.iloc[0]["trip_count"]

    def test_insufficient_history_dropped(self, fe: FeatureEngineer) -> None:
        """Rows with NaN lags (insufficient history) are dropped."""
        panel = _make_hourly_panel()
        result = fe.add_lag_features(panel)
        lag_cols = [f"trip_count_lag_{h}" for h in fe.lag_hours]
        assert result[lag_cols].isna().sum().sum() == 0

    def test_no_cross_zone_bleed(self, fe: FeatureEngineer) -> None:
        """Lags do not bleed across different zones."""
        panel = _make_hourly_panel()
        result = fe.add_lag_features(panel)

        for zone in result["pulocationid"].unique():
            zone_data = result[result["pulocationid"] == zone].sort_values("hour")
            for h in fe.lag_hours:
                shifted = zone_data["trip_count"].shift(h)
                valid = shifted.dropna()
                actual = zone_data.loc[valid.index, f"trip_count_lag_{h}"]
                pd.testing.assert_series_equal(
                    actual.reset_index(drop=True),
                    valid.astype(int).reset_index(drop=True),
                    check_names=False,
                )


# -- Test: add_zone_coordinates ---------------------------------------------


class TestAddZoneCoordinates:
    """Tests for FeatureEngineer.add_zone_coordinates."""

    def test_columns_added(self, fe: FeatureEngineer) -> None:
        """pickup_longitude and pickup_latitude columns are added."""
        panel = _make_hourly_panel()
        result = fe.add_zone_coordinates(panel)
        assert "pickup_longitude" in result.columns
        assert "pickup_latitude" in result.columns

    def test_row_count_unchanged(self, fe: FeatureEngineer) -> None:
        """Row count is unchanged after merge."""
        panel = _make_hourly_panel()
        result = fe.add_zone_coordinates(panel)
        assert len(result) == len(panel)

    def test_unknown_zone_dropped(self, fe: FeatureEngineer) -> None:
        """Rows with unknown zone IDs are dropped."""
        panel = pd.DataFrame(
            {
                "pulocationid": [99999],
                "hour": [pd.Timestamp("2022-01-01")],
                "trip_count": [5],
            }
        )
        result = fe.add_zone_coordinates(panel)
        assert len(result) == 0


# -- Test: run end-to-end --------------------------------------------------


class TestRunEndToEnd:
    """Tests for FeatureEngineer.run (full pipeline)."""

    def test_gold_output(self, tmp_path: Any, fe: FeatureEngineer) -> None:
        """Gold parquet is written with all expected columns."""
        silver = _make_silver_df()
        silver_path = str(tmp_path / "silver.parquet")
        gold_dir = str(tmp_path / "gold" / "yellow_tripdata")
        silver.to_parquet(silver_path, index=False)

        config = {
            **fe.config,
            "data": {
                **fe.config["data"],
                "silver_path": silver_path,
                "gold_path": gold_dir,
            },
        }
        eng = FeatureEngineer(config=config)
        result = eng.run()

        # File was written
        output_path = os.path.join(gold_dir, "hourly_demand_featured.parquet")
        assert os.path.exists(output_path)

        # Expected column groups present
        assert "trip_count" in result.columns
        assert "hour_of_day" in result.columns
        assert "hour_of_day_sin" in result.columns
        for h in fe.lag_hours:
            assert f"trip_count_lag_{h}" in result.columns
        assert "pickup_longitude" in result.columns

        # No NaN in lag columns
        lag_cols = [f"trip_count_lag_{h}" for h in fe.lag_hours]
        assert result[lag_cols].isna().sum().sum() == 0

        # Sorted by (zone, hour)
        sorted_result = result.sort_values(["pulocationid", "hour"]).reset_index(
            drop=True
        )
        pd.testing.assert_frame_equal(result.reset_index(drop=True), sorted_result)


# -- Test: add_neighbor_lag_smoothing (PBI B1) ------------------------------


def _make_neighbor_panel(zone_lag1: dict[int, int]) -> pd.DataFrame:
    """Build a 1-bucket panel with deterministic lag_1 values per zone.

    Zones are placed on a line: zone z at longitude -74.00 - 0.01*z, so
    zone 1 and zone 2 are the closest pair, zone 2 and zone 3 are next, etc.
    """
    rows: list[dict[str, object]] = []
    for z, lag1 in zone_lag1.items():
        rows.append(
            {
                "pulocationid": z,
                "hour": pd.Timestamp("2022-01-01 00:00"),
                "trip_count_lag_1": lag1,
                "pickup_longitude": -74.00 - 0.01 * z,
                "pickup_latitude": 40.70,
            }
        )
    return pd.DataFrame(rows)


def _fe_with_neighbors(k: int) -> FeatureEngineer:
    cfg: dict[str, Any] = {
        "env": "local",
        "data": {
            "silver_path": "data/silver/test.parquet",
            "gold_path": "data/gold/yellow_tripdata",
        },
        "feature": {
            "lags": [1],
            "n_neighbors": k,
            "zone_shapefile_path": "data/reference/taxi_zones.shp",
        },
    }
    return FeatureEngineer(config=cfg)


class TestNeighborLagSmoothing:
    """Tests for FeatureEngineer.add_neighbor_lag_smoothing (PBI B1)."""

    def test_neighbor_mean_matches_manual_calculation(self) -> None:
        """k=2 neighbour-mean on a known line matches hand-computed values."""
        # Zones on a line at increasing longitude. Lag_1 values per zone:
        # 1→10, 2→20, 3→30, 4→40, 5→50, 6→60
        panel = _make_neighbor_panel({1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60})
        result = _fe_with_neighbors(k=2).add_neighbor_lag_smoothing(panel)

        means = result.set_index("pulocationid")["trip_count_lag_1_neighbor_mean"]
        # Zone 1's two nearest neighbours are 2 and 3: mean(20, 30) = 25
        assert means.loc[1] == pytest.approx(25.0)
        # Zone 3's two nearest are 2 and 4: mean(20, 40) = 30
        assert means.loc[3] == pytest.approx(30.0)
        # Zone 6's two nearest are 5 and 4: mean(50, 40) = 45
        assert means.loc[6] == pytest.approx(45.0)

    def test_neighbor_mean_scale_preserved(self) -> None:
        """Neighbour-mean stays on the same count scale as the source lag.

        With every zone holding the same lag_1 value, the neighbour-mean
        for every zone must equal that value exactly — the "mean (not sum)"
        choice locked here.
        """
        panel = _make_neighbor_panel({z: 7 for z in range(1, 6)})
        result = _fe_with_neighbors(k=2).add_neighbor_lag_smoothing(panel)
        means = result["trip_count_lag_1_neighbor_mean"]
        assert (means == 7).all()

    def test_neighbor_mean_no_self_inclusion(self) -> None:
        """A zone's own lag value must not contribute to its own neighbour-mean.

        Zone 1 holds lag_1=100, every other zone holds 0. Zone 1's
        neighbour-mean must be 0 — if self were included it would be > 0.
        """
        panel = _make_neighbor_panel({1: 100, 2: 0, 3: 0, 4: 0, 5: 0})
        result = _fe_with_neighbors(k=2).add_neighbor_lag_smoothing(panel)
        means = result.set_index("pulocationid")["trip_count_lag_1_neighbor_mean"]
        assert means.loc[1] == 0

    def test_orphan_zones_dropped_upstream(self) -> None:
        """Zones missing centroids before this step yield NaN neighbour-mean.

        ``add_neighbor_lag_smoothing`` trusts ``add_zone_coordinates`` to
        have already dropped shapefile-orphan zones (e.g. 57, 264, 265).
        If a stray orphan slips through, the haversine query can't place
        it on the tree and the row must surface as NaN rather than silently
        get the global mean.
        """
        panel = _make_neighbor_panel({1: 10, 2: 20, 3: 30})
        # Inject an orphan: zone 99 with NaN coordinates
        orphan = pd.DataFrame(
            [
                {
                    "pulocationid": 99,
                    "hour": pd.Timestamp("2022-01-01 00:00"),
                    "trip_count_lag_1": 5,
                    "pickup_longitude": np.nan,
                    "pickup_latitude": np.nan,
                }
            ]
        )
        panel_with_orphan = pd.concat([panel, orphan], ignore_index=True)
        result = _fe_with_neighbors(k=2).add_neighbor_lag_smoothing(panel_with_orphan)
        means = result.set_index("pulocationid")["trip_count_lag_1_neighbor_mean"]
        # Zone 99 has NaN centroid → BallTree can't place it → NaN result
        assert pd.isna(means.loc[99])

    def test_n_neighbors_zero_is_noop(self) -> None:
        """``n_neighbors=0`` (legacy back-compat) must not modify the panel."""
        panel = _make_neighbor_panel({1: 10, 2: 20, 3: 30})
        result = _fe_with_neighbors(k=0).add_neighbor_lag_smoothing(panel)
        assert "trip_count_lag_1_neighbor_mean" not in result.columns
        pd.testing.assert_frame_equal(result, panel)


# -- Test: add_summed_horizon_target (PBI B2) -------------------------------


def _fe_with_horizon_sum(n: int) -> FeatureEngineer:
    cfg: dict[str, Any] = {
        "env": "local",
        "data": {
            "silver_path": "data/silver/test.parquet",
            "gold_path": "data/gold/yellow_tripdata",
        },
        "feature": {
            "lags": [1],
            "horizon_sum_buckets": n,
            "zone_shapefile_path": "data/reference/taxi_zones.shp",
        },
    }
    return FeatureEngineer(config=cfg)


class TestSummedHorizonTarget:
    """Tests for FeatureEngineer.add_summed_horizon_target (PBI B2)."""

    def test_sum_of_two_future_buckets(self) -> None:
        """target_sum_t1_t2 must equal shift(-1) + shift(-2) of trip_count."""
        panel = _make_hourly_panel().sort_values(["pulocationid", "hour"])
        result = _fe_with_horizon_sum(n=2).add_summed_horizon_target(panel)

        # For each (zone, hour) in result, the column must equal
        # trip_count[t+1] + trip_count[t+2] in the original panel.
        for zone in panel["pulocationid"].unique():
            orig = (
                panel[panel["pulocationid"] == zone]
                .sort_values("hour")
                .reset_index(drop=True)
            )
            out = (
                result[result["pulocationid"] == zone]
                .sort_values("hour")
                .reset_index(drop=True)
            )
            for i in range(len(out)):
                expected = int(
                    orig["trip_count"].iloc[i + 1] + orig["trip_count"].iloc[i + 2]
                )
                assert out["target_sum_t1_t2"].iloc[i] == expected, (
                    f"zone={zone} row={i}: "
                    f"got {out['target_sum_t1_t2'].iloc[i]}, expected {expected}"
                )

    def test_shift_direction_is_forward(self) -> None:
        """Anti-leakage guard: target must come from the FUTURE, not the past.

        Construct a strictly-increasing trip_count series. If the shift
        direction were wrong (past as target), the summed target would be
        *smaller* than trip_count. Forward shift guarantees it's *larger*.
        """
        zone_data = pd.DataFrame(
            {
                "pulocationid": [1] * 10,
                "hour": pd.date_range("2022-01-01", periods=10, freq="h"),
                "trip_count": list(range(1, 11)),  # 1, 2, 3, ..., 10
            }
        )
        result = _fe_with_horizon_sum(n=2).add_summed_horizon_target(zone_data)

        # For each retained row, target_sum > trip_count (future > present
        # when the underlying series is strictly increasing)
        assert (result["target_sum_t1_t2"] > result["trip_count"]).all()

    def test_no_cross_zone_bleed(self) -> None:
        """Summing must not pull values from a different zone."""
        # Zone 1: counts 1..5; Zone 2: counts 100..104. If summing leaked
        # across zones, zone 1's last retained row would include zone 2's
        # values and explode upward.
        df = pd.DataFrame(
            {
                "pulocationid": [1] * 5 + [2] * 5,
                "hour": list(pd.date_range("2022-01-01", periods=5, freq="h")) * 2,
                "trip_count": list(range(1, 6)) + list(range(100, 105)),
            }
        )
        result = _fe_with_horizon_sum(n=2).add_summed_horizon_target(df)

        zone1 = result[result["pulocationid"] == 1]
        # Zone 1's maximum summed target = trip_count[3] + trip_count[4] = 4+5 = 9.
        # If zone 2 leaked in, this would be 100+ instead.
        assert zone1["target_sum_t1_t2"].max() < 50

    def test_horizon_sum_zero_is_noop(self) -> None:
        """``horizon_sum_buckets=0`` must leave the panel untouched."""
        panel = _make_hourly_panel()
        result = _fe_with_horizon_sum(n=0).add_summed_horizon_target(panel)
        assert "target_sum_t1_t2" not in result.columns
        assert "target_sum_t1_t1" not in result.columns
        pd.testing.assert_frame_equal(result, panel)

    def test_trailing_rows_dropped(self) -> None:
        """The last N rows per zone (no future to sum) must be dropped."""
        n_per_zone = 10
        df = pd.DataFrame(
            {
                "pulocationid": [1] * n_per_zone + [2] * n_per_zone,
                "hour": list(pd.date_range("2022-01-01", periods=n_per_zone, freq="h"))
                * 2,
                "trip_count": [5] * (2 * n_per_zone),
            }
        )
        result = _fe_with_horizon_sum(n=2).add_summed_horizon_target(df)
        # With N=2, each zone loses its last 2 rows
        for zone in (1, 2):
            assert (result["pulocationid"] == zone).sum() == n_per_zone - 2


# -- Test: build_centroid_lookup -------------------------------------------


def _make_fake_geodataframe(
    location_ids: list[int],
    lons: list[float],
    lats: list[float],
    null_indices: list[int] | None = None,
) -> gpd.GeoDataFrame:
    """Build a minimal GeoDataFrame that mimics the taxi zones shapefile.

    Creates simple square polygons so that centroid computation can run
    without accessing real files on disk.

    Args:
        location_ids: List of integer zone IDs.
        lons: Approximate centroid longitude for each zone (WGS84).
        lats: Approximate centroid latitude for each zone (WGS84).
        null_indices: Indices (0-based) that should have null geometry.

    Returns:
        GeoDataFrame with columns ``LocationID`` and ``geometry`` in
        EPSG:4326.
    """
    null_indices = null_indices or []
    geometries = []
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        if i in null_indices:
            geometries.append(None)
        else:
            # 0.01° square centred at (lon, lat)
            half = 0.005
            geometries.append(
                Polygon(
                    [
                        (lon - half, lat - half),
                        (lon + half, lat - half),
                        (lon + half, lat + half),
                        (lon - half, lat + half),
                    ]
                )
            )
    gdf = gpd.GeoDataFrame(
        {"LocationID": location_ids, "geometry": geometries},
        crs="EPSG:4326",
    )
    return gdf


class TestBuildCentroidLookup:
    """Unit tests for the module-level ``build_centroid_lookup`` helper."""

    def test_build_centroid_lookup_returns_expected_keys(self) -> None:
        """Dict keys must match the LocationIDs in the shapefile.

        Arrange: fake GDF with zones 1, 2, 3.
        Act: call build_centroid_lookup with a mocked gpd.read_file.
        Assert: returned dict has exactly those three keys.
        """
        fake_gdf = _make_fake_geodataframe(
            location_ids=[1, 2, 3],
            lons=[-74.0, -73.9, -73.8],
            lats=[40.7, 40.8, 40.9],
        )
        with patch("app.ingestion.feature.gpd.read_file", return_value=fake_gdf):
            result = build_centroid_lookup("fake/path.shp")

        assert set(result.keys()) == {1, 2, 3}

    def test_build_centroid_lookup_values_rounded_to_3dp(self) -> None:
        """Each (lat, lon) pair must be rounded to exactly 3 decimal places.

        Arrange: fake GDF with a single zone at a coordinate that would
        produce more than 3 d.p. when computed naively.
        Act: call build_centroid_lookup.
        Assert: both latitude and longitude in the tuple have at most 3 d.p.
        """
        fake_gdf = _make_fake_geodataframe(
            location_ids=[10],
            lons=[-73.987654],
            lats=[40.712345],
        )
        with patch("app.ingestion.feature.gpd.read_file", return_value=fake_gdf):
            result = build_centroid_lookup("fake/path.shp")

        lat, lon = result[10]
        # Check that the values are rounded: multiply by 1000 and verify
        # the fractional part is zero (i.e. no digit beyond 3 d.p.).
        assert lat == round(lat, 3)
        assert lon == round(lon, 3)
        # The rounded value must differ from full-precision by at most 0.0005.
        assert abs(lat - 40.712345) <= 0.0005
        assert abs(lon - (-73.987654)) <= 0.0005

    def test_build_centroid_lookup_missing_zone_absent_from_dict(self) -> None:
        """Zones with null geometry must be absent from the returned dict.

        Arrange: fake GDF with zones 1, 2, 3 where zone 2 has null geometry.
        Act: call build_centroid_lookup.
        Assert: key 2 is not in the result; keys 1 and 3 are present.
        """
        fake_gdf = _make_fake_geodataframe(
            location_ids=[1, 2, 3],
            lons=[-74.0, -73.9, -73.8],
            lats=[40.7, 40.8, 40.9],
            null_indices=[1],  # zone 2 (index 1) has null geometry
        )
        with patch("app.ingestion.feature.gpd.read_file", return_value=fake_gdf):
            result = build_centroid_lookup("fake/path.shp")

        assert 2 not in result
        assert 1 in result
        assert 3 in result
