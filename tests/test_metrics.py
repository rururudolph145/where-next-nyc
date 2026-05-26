"""Tests for app/core/metrics.py — evaluate, compute_recall_at_k, compute_wape_lift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.metrics import compute_recall_at_k, compute_wape_lift, evaluate


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    """Tests for the canonical evaluate() function."""

    def test_evaluate_returns_expected_keys(self) -> None:
        """AC #6: evaluate returns exactly {mae, rmse, mape, wape}."""
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        result = evaluate(y_true, y_pred)

        assert set(result.keys()) == {"mae", "rmse", "mape", "wape"}

    def test_evaluate_all_zeros_wape_is_zero(self) -> None:
        """y_true all zeros → wape == 0.0, no divide-by-zero raised."""
        y_true = pd.Series([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        result = evaluate(y_true, y_pred)

        assert result["wape"] == 0.0

    def test_evaluate_perfect_predictions_all_metrics_zero(self) -> None:
        """Perfect predictions yield mae == rmse == mape == wape == 0.0."""
        y_true = pd.Series([5.0, 10.0, 15.0])
        y_pred = np.array([5.0, 10.0, 15.0])

        result = evaluate(y_true, y_pred)

        assert result["mae"] == pytest.approx(0.0)
        assert result["rmse"] == pytest.approx(0.0)
        assert result["mape"] == pytest.approx(0.0)
        assert result["wape"] == pytest.approx(0.0)

    def test_evaluate_known_values_mae(self) -> None:
        """MAE computation matches hand-calculated value."""
        y_true = pd.Series([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 33.0])
        # |10-12|=2, |20-18|=2, |30-33|=3 → mean=7/3
        expected_mae = (2 + 2 + 3) / 3

        result = evaluate(y_true, y_pred)

        assert result["mae"] == pytest.approx(expected_mae)


# ---------------------------------------------------------------------------
# compute_recall_at_k
# ---------------------------------------------------------------------------


class TestComputeRecallAtK:
    """Tests for compute_recall_at_k."""

    def _make_dfs(
        self,
        pred_ranks: list[list[tuple[int, float]]],
        actual_ranks: list[list[tuple[int, float]]],
        hours: list[pd.Timestamp] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build predicted/actual DataFrames from (zone, value) lists per bucket."""
        if hours is None:
            hours = [
                pd.Timestamp(f"2022-01-01 {i:02d}:00") for i in range(len(pred_ranks))
            ]
        pred_rows = []
        actual_rows = []
        for h, pred_bucket, actual_bucket in zip(hours, pred_ranks, actual_ranks):
            for z, v in pred_bucket:
                pred_rows.append({"pulocationid": z, "hour": h, "value": v})
            for z, v in actual_bucket:
                actual_rows.append({"pulocationid": z, "hour": h, "value": v})
        return pd.DataFrame(pred_rows), pd.DataFrame(actual_rows)

    def test_compute_recall_at_k_manual_fixture(self) -> None:
        """AC #4: hand-computed recall for 3 zones × 2 buckets matches expected."""
        # Bucket 0: pred top-3=[1,2,3], actual top-3=[1,2,4] → overlap={1,2}=2 → recall=2/3
        # Bucket 1: pred top-3=[1,2,3], actual top-3=[1,2,3] → overlap={1,2,3}=3 → recall=3/3=1
        # Mean recall = (2/3 + 1.0) / 2 = 5/6 ≈ 0.8333
        pred_b0 = [(1, 30.0), (2, 20.0), (3, 10.0), (4, 5.0)]
        act_b0 = [(1, 30.0), (2, 20.0), (4, 15.0), (3, 5.0)]
        pred_b1 = [(1, 30.0), (2, 20.0), (3, 10.0)]
        act_b1 = [(1, 30.0), (2, 20.0), (3, 10.0)]

        predicted, actual = self._make_dfs([pred_b0, pred_b1], [act_b0, act_b1])
        result = compute_recall_at_k(predicted, actual, k=3)

        assert result == pytest.approx(5 / 6, abs=1e-9)

    def test_compute_recall_at_k_all_zones_tied(self) -> None:
        """AC #5: ties broken by ascending pulocationid; top-k is deterministic."""
        # All predicted values equal — stable sort picks zones 1, 2, 3
        # Actual top-3 = zones 1, 2, 3 → recall should be 1.0
        h = pd.Timestamp("2022-01-01")
        pred_rows = [{"pulocationid": z, "hour": h, "value": 5.0} for z in [3, 1, 4, 2]]
        act_rows = [
            {"pulocationid": z, "hour": h, "value": float(z)} for z in [1, 2, 3, 4]
        ]
        predicted = pd.DataFrame(pred_rows)
        actual = pd.DataFrame(act_rows)

        result = compute_recall_at_k(predicted, actual, k=3)

        # pred top-3 by (value desc, pulocationid asc) = zones 1, 2, 3
        # actual top-3 by (value desc, pulocationid asc) = zones 4, 3, 2
        # overlap = {2, 3} → recall = 2/3
        assert result == pytest.approx(2 / 3, abs=1e-9)

    def test_compute_recall_at_k_fewer_than_k_zones(self) -> None:
        """Bucket with 2 zones and k=3: recall ≤ 1.0, denominator stays 3."""
        h = pd.Timestamp("2022-01-01")
        pred_rows = [
            {"pulocationid": 1, "hour": h, "value": 10.0},
            {"pulocationid": 2, "hour": h, "value": 5.0},
        ]
        act_rows = [
            {"pulocationid": 1, "hour": h, "value": 10.0},
            {"pulocationid": 2, "hour": h, "value": 5.0},
        ]
        predicted = pd.DataFrame(pred_rows)
        actual = pd.DataFrame(act_rows)

        result = compute_recall_at_k(predicted, actual, k=3)

        # Best case: overlap = {1, 2} = 2 zones → recall = 2/3 (denom stays 3)
        assert result == pytest.approx(2 / 3, abs=1e-9)
        assert result <= 1.0


# ---------------------------------------------------------------------------
# compute_wape_lift
# ---------------------------------------------------------------------------


class TestComputeWapeLift:
    """Tests for compute_wape_lift."""

    def test_compute_wape_lift_positive(self) -> None:
        """AC #2: baseline_wape=40.0, model_wape=30.0 → lift=25.0."""
        result = compute_wape_lift(baseline_wape=40.0, model_wape=30.0)

        assert result == pytest.approx(25.0)

    def test_compute_wape_lift_negative_lift(self) -> None:
        """AC #2: model worse than baseline → negative lift returned as-is."""
        result = compute_wape_lift(baseline_wape=30.0, model_wape=35.0)

        assert result < 0.0
        assert result == pytest.approx(-100 * 5 / 30)

    def test_compute_wape_lift_baseline_zero_returns_inf(self) -> None:
        """baseline_wape == 0.0 → float('inf') to guard divide-by-zero."""
        result = compute_wape_lift(baseline_wape=0.0, model_wape=10.0)

        assert result == float("inf")
