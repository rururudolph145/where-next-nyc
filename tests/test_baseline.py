"""Tests for app/training/baseline.py — LastWeekSameHourBaseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.training.baseline import LastWeekSameHourBaseline


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _make_baseline_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build 4 weeks × 24 hours × 3 zones of synthetic Gold data.

    Pattern: trip_count = zone_id * 10 + hour_of_day (deterministic, weekly repeat).

    Returns:
        (gold_df, test_df) where test_df is the last week of gold_df.
    """
    n_weeks = 4
    n_hours = 24
    zones = [1, 2, 3]

    records = []
    base = pd.Timestamp("2022-01-01 00:00:00")
    for week in range(n_weeks):
        for hour in range(n_hours):
            ts = base + pd.Timedelta(weeks=week, hours=hour)
            for z in zones:
                records.append(
                    {
                        "pulocationid": z,
                        "hour": ts,
                        "trip_count": float(z * 10 + hour),
                    }
                )

    gold_df = pd.DataFrame(records)
    # test_df is the last week
    cutoff = base + pd.Timedelta(weeks=3)
    test_df = gold_df[gold_df["hour"] >= cutoff].copy().reset_index(drop=True)
    return gold_df, test_df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLastWeekSameHourBaseline:
    """Tests for LastWeekSameHourBaseline.predict."""

    def test_predict_returns_exact_week_ago_count_for_known_rows(self) -> None:
        """AC #3: y_pred_baseline matches the gold count from one week earlier."""
        gold_df, test_df = _make_baseline_fixture()
        baseline = LastWeekSameHourBaseline(target_col="trip_count", bucket="h")

        result = baseline.predict(test_df, gold_df)

        # For each row in result, the pred should equal trip_count(zone, hour-7d)
        for _, row in result.iterrows():
            week_ago = row["hour"] - pd.Timedelta(days=7)
            expected = gold_df.loc[
                (gold_df["pulocationid"] == row["pulocationid"])
                & (gold_df["hour"] == week_ago),
                "trip_count",
            ].values[0]
            assert row["y_pred_baseline"] == expected

    def test_predict_aligned_frame_columns(self) -> None:
        """Returned DataFrame has exactly the expected column names."""
        gold_df, test_df = _make_baseline_fixture()
        baseline = LastWeekSameHourBaseline(target_col="trip_count", bucket="h")

        result = baseline.predict(test_df, gold_df)

        assert list(result.columns) == [
            "pulocationid",
            "hour",
            "y_true",
            "y_pred_baseline",
        ]

    def test_predict_drops_rows_where_week_ago_missing(self) -> None:
        """Rows whose week-ago timestamp is absent in gold_df are excluded."""
        gold_df, _ = _make_baseline_fixture()
        baseline = LastWeekSameHourBaseline(target_col="trip_count", bucket="h")

        # Use only the FIRST week as test — no week-ago data exists in gold
        base = pd.Timestamp("2022-01-01 00:00:00")
        first_week_test = gold_df[gold_df["hour"] < base + pd.Timedelta(weeks=1)].copy()

        result = baseline.predict(first_week_test, gold_df)

        assert len(result) == 0

    def test_predict_week_offset_uses_timedelta_not_bucket_integer(self) -> None:
        """With bucket='30min', join uses pd.Timedelta(days=7) not bucket-integer shift.

        If bucket-integer math (7 * 24 * 2 = 336 rows back) were used, the
        result would diverge from a direct 7-day wall-clock subtraction when
        the test and gold DataFrames are asymmetrically padded.
        """
        # Gold: 2 weeks of 30-min buckets for zone 1
        base = pd.Timestamp("2022-01-01 00:00:00")
        n_buckets = 2 * 7 * 24 * 2  # 2 weeks × 30-min buckets
        hours = pd.date_range(base, periods=n_buckets, freq="30min")
        gold_df = pd.DataFrame(
            {
                "pulocationid": 1,
                "hour": hours,
                "trip_count": np.arange(n_buckets, dtype=float),
            }
        )

        # Test: use the second week
        week_2_start = base + pd.Timedelta(weeks=1)
        test_df = gold_df[gold_df["hour"] >= week_2_start].copy().reset_index(drop=True)

        baseline = LastWeekSameHourBaseline(target_col="trip_count", bucket="30min")
        result = baseline.predict(test_df, gold_df)

        assert len(result) > 0

        # Each predicted value must equal the gold value exactly 7 days earlier
        for _, row in result.iterrows():
            week_ago_ts = row["hour"] - pd.Timedelta(days=7)
            expected_val = gold_df.loc[
                (gold_df["pulocationid"] == row["pulocationid"])
                & (gold_df["hour"] == week_ago_ts),
                "trip_count",
            ].values[0]
            assert row["y_pred_baseline"] == pytest.approx(expected_val)
