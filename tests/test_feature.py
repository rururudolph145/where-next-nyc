"""Tests for the feature engineering module (app/ingestion/feature.py)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.ingestion.feature import FeatureEngineer

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
