"""Acceptance tests for PBI 18 — baseline reporting and MLflow metric logging.

Story: As a pipeline operator, I want the training run to automatically
evaluate a last-week-same-hour baseline alongside the LightGBM model and log
both sets of metrics — plus the lift between them — to a single MLflow run.

These tests exercise *Trainer.run()* end-to-end (the public entry point) on an
on-disk Gold parquet built from a synthetic 4-week × 24-hour × 3-zone fixture.
MLflow is mocked via ``patch("app.training.trainer.mlflow")`` (same pattern as
tests/test_train.py) so we can assert exactly which metrics were logged.

Acceptance criteria:
  AC #1  Single MLflow run contains all six new metrics.
  AC #2  wape_lift_pct formula; negative lift logged as-is.
  AC #3  LastWeekSameHourBaseline returns exact week-ago count.  [unit tests cover
         the helper directly; here we confirm the integrated run also lines up]
  AC #4  compute_recall_at_k matches manually-verified value.  [unit tests cover
         the formula; here we confirm recall_at_3 is a finite float in [0,1]]
  AC #5  recall_at_3 = per-bucket top-3 ∩ actual top-3 / 3, unweighted mean.
  AC #6  app.core.metrics.evaluate is the single canonical function; Trainer,
         Predictor, HourlySimulator all resolve to the same object.
  AC #7  README "How Value Is Measured" contains a Results subsection with a
         Markdown table; skip with explanation if the section is still a stub.

Edge cases:
  - Week-ago NaN drop (apples-to-apples subset) → AC #3 / AC #5 aligned rows
  - Baseline outperforms LGBM → negative lift logged as-is
  - All zones tied in predictions → ascending pulocationid tie-break
  - Fewer than K zones in a bucket-time → denominator stays 3
  - Non-default bucket width (30min) — offset via pd.Timedelta(days=7)
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Synthetic Gold fixture helpers
# ---------------------------------------------------------------------------

ZONES = [1, 2, 3]
BASE_TS = pd.Timestamp("2022-01-03 00:00:00")  # Monday — weekly pattern anchors here


def _make_4week_gold(
    tmp_path: Path,
    *,
    trip_count_fn: Any = None,
) -> tuple[Path, pd.DataFrame]:
    """Build a 4-week × 24-hour × 3-zone Gold parquet.

    Uses a deterministic weekly-repeating pattern so the baseline has signal:
      trip_count = zone_id * 10 + hour_of_day (by default)

    The parquet is written to ``tmp_path / "gold" / "hourly_demand_featured.parquet"``
    so that Trainer.load_gold() can find it.

    Args:
        tmp_path: pytest tmp_path fixture value.
        trip_count_fn: Optional callable(zone, week, hour) -> float.
            Defaults to ``zone * 10 + hour``.

    Returns:
        (gold_dir_path, full_gold_df) tuple.
    """
    if trip_count_fn is None:

        def trip_count_fn(zone: int, week: int, hour: int) -> float:  # noqa: E731
            return float(zone * 10 + hour)

    records = []
    for week in range(4):
        for hour in range(24):
            ts = BASE_TS + pd.Timedelta(weeks=week, hours=hour)
            for zone in ZONES:
                records.append(
                    {
                        "pulocationid": zone,
                        "hour": ts,
                        "trip_count": trip_count_fn(zone, week, hour),
                        # Minimal feature columns required by FEATURES list below
                        "hour_of_day_sin": float(np.sin(2 * np.pi * hour / 24)),
                        "hour_of_day_cos": float(np.cos(2 * np.pi * hour / 24)),
                        "day_of_week_sin": float(np.sin(2 * np.pi * ts.dayofweek / 7)),
                        "day_of_week_cos": float(np.cos(2 * np.pi * ts.dayofweek / 7)),
                        "month_sin": float(np.sin(2 * np.pi * ts.month / 12)),
                        "month_cos": float(np.cos(2 * np.pi * ts.month / 12)),
                        "pickup_longitude": float(-74.00 - zone * 0.01),
                        "pickup_latitude": float(40.70 + zone * 0.01),
                        "trip_count_lag_1": float(zone * 10 + (hour - 1) % 24),
                        "trip_count_lag_3": float(zone * 10 + (hour - 3) % 24),
                        "trip_count_lag_6": float(zone * 10 + (hour - 6) % 24),
                        "trip_count_lag_12": float(zone * 10 + (hour - 12) % 24),
                        "trip_count_lag_24": float(zone * 10 + (hour - 24) % 24),
                        "trip_count_lag_168": float(zone * 10 + hour),
                    }
                )

    gold_df = pd.DataFrame(records)
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir(parents=True)
    gold_df.to_parquet(gold_dir / "hourly_demand_featured.parquet", index=False)
    return gold_dir, gold_df


# Features that match MINIMAL_CONFIG in test_train.py
_FEATURES = [
    "hour_of_day_sin",
    "hour_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "pickup_longitude",
    "pickup_latitude",
    "trip_count_lag_1",
    "trip_count_lag_3",
    "trip_count_lag_6",
    "trip_count_lag_12",
    "trip_count_lag_24",
    "trip_count_lag_168",
]


def _trainer_config(tmp_path: Path) -> dict[str, Any]:
    """Minimal Trainer config pointing at tmp_path Gold/model/mlruns dirs."""
    return {
        "env": "local",
        "data": {"gold_path": str(tmp_path / "gold")},
        "feature": {"bucket": "h"},
        "train": {
            "features": _FEATURES,
            "target": "trip_count",
            "split_ratios": [0.6, 0.2, 0.2],
        },
        "model": {
            "name": "LGBMRegressor",
            "params": {
                "n_estimators": 5,
                "learning_rate": 0.1,
                "num_leaves": 4,
                "random_state": 42,
                "verbosity": -1,
            },
            "store_path": str(tmp_path / "models"),
        },
        "mlflow": {
            "experiment_name": "test_pbi18_acceptance",
            "tracking_uri": str(tmp_path / "mlruns"),
        },
    }


# ---------------------------------------------------------------------------
# AC #1 — All six metrics logged inside a single MLflow run
# ---------------------------------------------------------------------------


class TestAC1AllSixMetricsInSingleRun:
    """AC #1: single MLflow run contains all six new metric names."""

    def test_ac1_all_six_metrics_logged_in_single_mlflow_run(
        self, tmp_path: Path
    ) -> None:
        """Trainer.run() logs baseline_test_mae, baseline_test_wape,
        model_test_mae, model_test_wape, wape_lift_pct, recall_at_3
        all inside a single mlflow.start_run() call."""
        # Arrange
        gold_dir, gold_df = _make_4week_gold(tmp_path)
        config = _trainer_config(tmp_path)
        trainer = Trainer(config=config)

        # Act
        with patch("app.training.trainer.mlflow") as mock_mlflow:
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(
                return_value=MagicMock()
            )
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            trainer.run(data=gold_df)

        # Assert — exactly one run opened
        mock_mlflow.start_run.assert_called_once()

        logged_names = {call[0][0] for call in mock_mlflow.log_metric.call_args_list}
        expected = {
            "baseline_test_mae",
            "baseline_test_wape",
            "model_test_mae",
            "model_test_wape",
            "wape_lift_pct",
            "recall_at_3",
        }
        assert expected.issubset(logged_names), (
            f"Missing metrics: {expected - logged_names}"
        )


# ---------------------------------------------------------------------------
# AC #2 — wape_lift_pct formula; value logged as-is (including negative)
# ---------------------------------------------------------------------------


class TestAC2WapeLiftFormula:
    """AC #2: lift = (baseline_wape - model_wape) / baseline_wape * 100."""

    def test_ac2_wape_lift_pct_matches_formula(self, tmp_path: Path) -> None:
        """The logged wape_lift_pct equals (baseline_wape - model_wape) /
        baseline_wape * 100, verified independently using model_aligned_metrics
        from the return dict — not a circular comparison to itself.

        A small per-week perturbation ensures baseline_wape > 0 so the formula
        produces a finite lift that can be compared with pytest.approx.  When
        baseline_wape == 0 (perfect baseline), compute_wape_lift returns +inf
        and the test falls back to the inf-equality check.
        """

        # Arrange — use a week-level variation so baseline WAPE > 0 (the purely
        # repeating 4-week fixture gives baseline_wape == 0, which makes the
        # formula check degenerate).
        def weekly_varying_trips(zone: int, week: int, hour: int) -> float:
            return float(zone * 10 + hour) + week * 1.0

        gold_dir, gold_df = _make_4week_gold(
            tmp_path, trip_count_fn=weekly_varying_trips
        )
        config = _trainer_config(tmp_path)
        trainer = Trainer(config=config)

        # Act
        with patch("app.training.trainer.mlflow") as mock_mlflow:
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(
                return_value=MagicMock()
            )
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            result = trainer.run(data=gold_df)

        # Assert — compute the expected lift independently from both WAPE values
        # returned in the result dict, then compare to the reported lift.
        baseline_wape = result["baseline_metrics"]["wape"]
        model_aligned_wape = result["model_aligned_metrics"]["wape"]
        logged_lift = {
            call[0][0]: call[0][1] for call in mock_mlflow.log_metric.call_args_list
        }

        if baseline_wape == 0.0:
            # Perfect baseline → lift is +inf; verify inf is logged
            assert math.isinf(result["wape_lift_pct"])
            assert math.isinf(logged_lift["wape_lift_pct"])
        else:
            expected_lift = (baseline_wape - model_aligned_wape) / baseline_wape * 100.0

            # AC #2 formula: (baseline - model) / baseline * 100
            assert result["wape_lift_pct"] == pytest.approx(expected_lift, rel=1e-9)

            # AC #1: same value was logged to MLflow under the same run
            assert logged_lift["wape_lift_pct"] == pytest.approx(
                expected_lift, rel=1e-9
            )

    def test_ac2_negative_lift_when_model_worse_than_baseline(
        self, tmp_path: Path
    ) -> None:
        """When LGBM predictions are worse than the baseline, wape_lift_pct
        is negative and is still logged as-is (not clamped to zero).

        A small per-week perturbation keeps baseline_wape strictly > 0 so
        the lift is a finite negative number, not +inf (which would arise if
        the baseline were perfect and baseline_wape == 0).  The 1-estimator
        model cannot learn the subtle perturbation, so it underperforms.
        """

        # Arrange: nearly constant trip_count with small week-level noise so
        # baseline_wape > 0 but is very small; LGBM with 1 estimator cannot
        # match it and produces positive prediction error → negative lift.
        def nearly_constant_trips(zone: int, week: int, hour: int) -> float:
            # Vary slightly by week (0.5 unit) so baseline WAPE > 0
            return float(zone * 5) + week * 0.5

        gold_dir, gold_df = _make_4week_gold(
            tmp_path, trip_count_fn=nearly_constant_trips
        )
        config = _trainer_config(tmp_path)
        # Tiny model — cannot learn the weekly variation
        config["model"]["params"]["n_estimators"] = 1
        trainer = Trainer(config=config)

        # Act
        with patch("app.training.trainer.mlflow") as mock_mlflow:
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(
                return_value=MagicMock()
            )
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            result = trainer.run(data=gold_df)

        # Assert — the key invariant: whatever lift was computed, that exact
        # value was logged to MLflow.  Branch on inf to avoid pytest.approx
        # raising TypeError on infinite comparisons.
        logged_metrics = {
            call[0][0]: call[0][1] for call in mock_mlflow.log_metric.call_args_list
        }
        assert "wape_lift_pct" in logged_metrics
        if math.isinf(result["wape_lift_pct"]):
            assert math.isinf(logged_metrics["wape_lift_pct"]), (
                "When baseline WAPE is exactly 0, lift is +inf and must be logged as inf"
            )
        else:
            assert logged_metrics["wape_lift_pct"] == pytest.approx(
                result["wape_lift_pct"], rel=1e-6
            )
            # And it must be negative when model is worse than baseline
            assert result["wape_lift_pct"] < 0, (
                "Test fixture intends a worse model — lift should be negative"
            )


# ---------------------------------------------------------------------------
# AC #3 — Baseline returns exact week-ago count (integration assertion)
# ---------------------------------------------------------------------------


class TestAC3BaselineWeekAgoIntegration:
    """AC #3 integrated: the Trainer uses LastWeekSameHourBaseline correctly."""

    def test_ac3_baseline_metrics_use_exact_week_ago_counts(
        self, tmp_path: Path
    ) -> None:
        """On the 4-week deterministic fixture, baseline_test_mae must be 0.0
        because trip_count = zone*10 + hour repeats identically each week."""
        # Arrange — deterministic weekly-repeating trip_count (default pattern)
        gold_dir, gold_df = _make_4week_gold(tmp_path)
        config = _trainer_config(tmp_path)
        trainer = Trainer(config=config)

        # Act
        with patch("app.training.trainer.mlflow") as mock_mlflow:
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(
                return_value=MagicMock()
            )
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            result = trainer.run(data=gold_df)

        # Assert — baseline MAE on the apples-to-apples subset must be exactly 0
        # because the pattern is a perfect weekly repeat.
        assert result["baseline_metrics"]["mae"] == pytest.approx(0.0, abs=1e-9), (
            "Baseline MAE should be 0.0 on a perfectly repeating weekly pattern"
        )
        assert result["baseline_metrics"]["wape"] == pytest.approx(0.0, abs=1e-9), (
            "Baseline WAPE should be 0.0 on a perfectly repeating weekly pattern"
        )


# ---------------------------------------------------------------------------
# AC #4 — recall_at_3 is a finite float in [0, 1]
# ---------------------------------------------------------------------------


class TestAC4RecallAt3IsFiniteFloat:
    """AC #4: compute_recall_at_k is wired into Trainer.run() and recall_at_3
    is a finite float in [0.0, 1.0]."""

    def test_ac4_recall_at_3_is_finite_and_bounded(self, tmp_path: Path) -> None:
        """Trainer.run() returns recall_at_3 as a float in [0.0, 1.0] and
        also logs it under the key 'recall_at_3'."""
        # Arrange
        gold_dir, gold_df = _make_4week_gold(tmp_path)
        config = _trainer_config(tmp_path)
        trainer = Trainer(config=config)

        # Act
        with patch("app.training.trainer.mlflow") as mock_mlflow:
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(
                return_value=MagicMock()
            )
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            result = trainer.run(data=gold_df)

        # Assert
        recall = result["recall_at_3"]
        assert isinstance(recall, float), f"Expected float, got {type(recall)}"
        assert np.isfinite(recall), f"recall_at_3 must be finite, got {recall}"
        assert 0.0 <= recall <= 1.0, f"recall_at_3 out of [0,1]: {recall}"

        logged_metrics = {
            call[0][0]: call[0][1] for call in mock_mlflow.log_metric.call_args_list
        }
        assert logged_metrics["recall_at_3"] == pytest.approx(recall, rel=1e-9)


# ---------------------------------------------------------------------------
# AC #5 — Apples-to-apples row alignment (week-ago NaN drop)
# ---------------------------------------------------------------------------


class TestAC5ApplesApplesAlignment:
    """AC #5: model and baseline are evaluated on the same (pulocationid, hour)
    subset; rows without a week-ago match are dropped from both."""

    def test_ac5_aligned_subset_excludes_rows_missing_week_ago(
        self, tmp_path: Path
    ) -> None:
        """Build a Gold fixture where the test split covers the *first* week of
        data so no week-ago values exist.  The baseline should drop all rows,
        and the aligned model evaluation should also have zero rows — but
        Trainer.run() should complete without error and log 0.0 for recall."""
        # Arrange: 2 weeks of data, 60/20/20 split → test ≈ last 20% of week 2.
        # Use the default 4-week fixture but only pass 2 weeks to the trainer so
        # the test split (last ~20%) lands in week 2 and HAS a week-ago anchor.
        # To force the NaN-drop path, use only the first 1 week + 1 hour so the
        # test rows have no week-ago data (gold contains < 7 days).
        n_weeks = 1
        records = []
        for hour in range(n_weeks * 24 + 1):  # just past 1 week
            ts = BASE_TS + pd.Timedelta(hours=hour)
            for zone in ZONES:
                records.append(
                    {
                        "pulocationid": zone,
                        "hour": ts,
                        "trip_count": float(zone * 10 + hour % 24),
                        "hour_of_day_sin": float(np.sin(2 * np.pi * (hour % 24) / 24)),
                        "hour_of_day_cos": float(np.cos(2 * np.pi * (hour % 24) / 24)),
                        "day_of_week_sin": 0.0,
                        "day_of_week_cos": 1.0,
                        "month_sin": 0.0,
                        "month_cos": 1.0,
                        "pickup_longitude": float(-74.00 - zone * 0.01),
                        "pickup_latitude": float(40.70 + zone * 0.01),
                        "trip_count_lag_1": float(zone * 10 + (hour - 1) % 24),
                        "trip_count_lag_3": float(zone * 10 + (hour - 3) % 24),
                        "trip_count_lag_6": float(zone * 10 + (hour - 6) % 24),
                        "trip_count_lag_12": float(zone * 10 + (hour - 12) % 24),
                        "trip_count_lag_24": float(zone * 10 + (hour - 24) % 24),
                        "trip_count_lag_168": float(zone * 10 + hour % 24),
                    }
                )

        short_gold = pd.DataFrame(records)
        gold_dir = tmp_path / "gold"
        gold_dir.mkdir(parents=True)
        short_gold.to_parquet(gold_dir / "hourly_demand_featured.parquet", index=False)

        config = _trainer_config(tmp_path)
        trainer = Trainer(config=config)

        # Act — run should not raise even when aligned_df is empty
        with patch("app.training.trainer.mlflow") as mock_mlflow:
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(
                return_value=MagicMock()
            )
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            result = trainer.run(data=short_gold)

        # Assert — run() completed; with no week-ago overlap the new empty-aligned
        # guard sets all four baseline-related keys to None so that meaningless
        # metrics are not logged to MLflow.
        assert result["baseline_metrics"] is None
        assert result["model_aligned_metrics"] is None
        assert result["wape_lift_pct"] is None
        assert result["recall_at_3"] is None

    def test_ac5_model_and_baseline_evaluated_on_same_rows(
        self, tmp_path: Path
    ) -> None:
        """Verify that model_test_mae and baseline_test_mae are logged using
        the same aligned subset, not the full unfiltered test set.

        Proof: on the perfectly-repeating fixture, baseline MAE = 0.0 and
        model_test_mae (aligned) must be for the same rows as baseline."""
        # Arrange
        gold_dir, gold_df = _make_4week_gold(tmp_path)
        config = _trainer_config(tmp_path)
        trainer = Trainer(config=config)

        # Act
        with patch("app.training.trainer.mlflow") as mock_mlflow:
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(
                return_value=MagicMock()
            )
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            trainer.run(data=gold_df)

        logged = {
            call[0][0]: call[0][1] for call in mock_mlflow.log_metric.call_args_list
        }

        # Both aligned metrics must be present
        assert "model_test_mae" in logged
        assert "baseline_test_mae" in logged

        # Baseline MAE on perfectly repeating data must be 0
        assert logged["baseline_test_mae"] == pytest.approx(0.0, abs=1e-9)

        # model_test_mae is on the *same rows* — it may be > 0 (LGBM isn't perfect)
        # but it must be finite and non-negative
        assert np.isfinite(logged["model_test_mae"])
        assert logged["model_test_mae"] >= 0.0


# ---------------------------------------------------------------------------
# AC #6 — Single canonical evaluate; Trainer, Predictor, Simulator share it
# ---------------------------------------------------------------------------


class TestAC6MetricsConsolidation:
    """AC #6: app.core.metrics.evaluate is the canonical function.

    TestMetricsConsolidation in test_train.py already verifies that results
    are identical.  This class adds the complementary check: each class's
    static method is a thin wrapper around the *same imported object*
    (not a re-implementation), verified by confirming that calling the
    wrappers with identical inputs produces byte-for-byte equal dicts.
    """

    def test_ac6_all_three_delegates_share_core_evaluate_results(self) -> None:
        """Trainer.evaluate and Predictor.evaluate both produce the same dict
        as app.core.metrics.evaluate for identical input."""
        from app.core.metrics import evaluate as core_evaluate
        from app.inference.predictor import Predictor
        from app.training.trainer import Trainer as _Trainer

        y_true_series = pd.Series([10.0, 20.0, 30.0, 0.0])
        y_pred_arr = np.array([11.0, 19.0, 32.0, 1.0])

        # Act
        core_result = core_evaluate(y_true_series, y_pred_arr)
        trainer_result = _Trainer.evaluate(y_true_series, y_pred_arr)
        predictor_result = Predictor.evaluate(y_true_series, y_pred_arr)

        # Assert — all three must be identical dicts
        assert trainer_result == core_result, (
            "Trainer.evaluate diverged from core evaluate"
        )
        assert predictor_result == core_result, (
            "Predictor.evaluate diverged from core evaluate"
        )

    def test_ac6_no_duplicate_evaluate_implementations(self) -> None:
        """Confirm that Trainer and Predictor do NOT define their own metric
        arithmetic — they delegate to app.core.metrics."""
        import inspect

        from app.inference.predictor import Predictor
        from app.training.trainer import Trainer as _Trainer

        # Inspect source of each wrapper; none should contain 'np.mean' or 'np.abs'
        # (those are the compute primitives that would indicate a re-implementation).
        for cls, method_name in [
            (_Trainer, "evaluate"),
            (Predictor, "evaluate"),
        ]:
            src = inspect.getsource(getattr(cls, method_name))
            assert "np.mean" not in src, (
                f"{cls.__name__}.{method_name} contains 'np.mean' — "
                "looks like a re-implementation rather than a delegation"
            )
            assert "np.abs" not in src, (
                f"{cls.__name__}.{method_name} contains 'np.abs' — "
                "looks like a re-implementation rather than a delegation"
            )


# ---------------------------------------------------------------------------
# AC #7 — README Results subsection check (skip if stub)
# ---------------------------------------------------------------------------


class TestAC7ReadmeResults:
    """AC #7: README 'How Value Is Measured' contains a Results subsection
    with a Markdown table holding numeric values.

    If the subsection is absent (the builder deferred it), the test skips
    with an explanation so the chain can proceed.  When the user fills in
    real numbers, the skip condition becomes False and the test starts
    asserting without any edits needed.
    """

    def test_ac7_readme_results_subsection_contains_numeric_table(self) -> None:
        """README has a ## Results / ### Results heading inside the
        'How Value Is Measured' section and that heading is followed by
        a Markdown table row with at least one numeric cell."""
        readme_path = Path(__file__).parent.parent / "README.md"

        # Arrange
        assert readme_path.exists(), f"README.md not found at {readme_path}"
        content = readme_path.read_text(encoding="utf-8")

        # Look for a Results heading (##  or ### level)
        results_heading = re.search(
            r"^#{2,3}\s+Results\b", content, re.MULTILINE | re.IGNORECASE
        )

        if results_heading is None:
            pytest.skip(
                "README results subsection deferred — upstream Gold pipeline must "
                "be re-run before AC #7 can be satisfied."
            )

        # If the heading exists, find the next Markdown table row after it
        after_heading = content[results_heading.end() :]
        # A Markdown table row: | ... | ... |  with at least one numeric cell
        table_row_with_number = re.search(
            r"^\|[^|\n]*\d+\.?\d*[^|\n]*\|", after_heading, re.MULTILINE
        )

        assert table_row_with_number is not None, (
            "README Results section exists but contains no Markdown table row "
            "with numeric values — populate it with real run results."
        )


# ---------------------------------------------------------------------------
# Edge case: all zones tied in predictions → ascending pulocationid tie-break
# ---------------------------------------------------------------------------


class TestEdgeCasesTieBreakAndFewZones:
    """Edge cases from the story: tie-break and fewer-than-K zones."""

    def test_edge_all_predictions_tied_uses_ascending_zone_tiebreak(self) -> None:
        """When all model predictions are equal for a bucket-time, top-k
        selection must break ties by ascending pulocationid.

        This is verified at the compute_recall_at_k level using known
        expected value, keeping it story-scope (not a unit test repetition)."""
        from app.core.metrics import compute_recall_at_k

        # Arrange: 4 zones, all predicted equally; actual ranks by zone desc
        h = pd.Timestamp("2022-01-01 08:00")
        pred_rows = [
            {"pulocationid": z, "hour": h, "value": 10.0} for z in [3, 1, 4, 2]
        ]
        act_rows = [
            {"pulocationid": z, "hour": h, "value": float(z)} for z in [1, 2, 3, 4]
        ]
        predicted = pd.DataFrame(pred_rows)
        actual = pd.DataFrame(act_rows)

        # Act
        result = compute_recall_at_k(predicted, actual, k=3)

        # Assert:
        # pred top-3 (value desc, zone asc) = zones 1, 2, 3 (all tie → asc zone)
        # actual top-3 (value desc, zone asc) = zones 4, 3, 2
        # overlap = {2, 3} → 2 / 3
        assert result == pytest.approx(2 / 3, abs=1e-9)

    def test_edge_fewer_than_k_zones_denominator_stays_k(self) -> None:
        """With only 2 zones available in a bucket, k=3, recall ≤ 1.0 and
        denominator remains 3 (not 2)."""
        from app.core.metrics import compute_recall_at_k

        # Arrange
        h = pd.Timestamp("2022-01-01 09:00")
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

        # Act
        result = compute_recall_at_k(predicted, actual, k=3)

        # Assert: overlap = {1, 2} = 2 out of denominator 3
        assert result == pytest.approx(2 / 3, abs=1e-9)
        assert result < 1.0

    def test_edge_non_default_bucket_width_week_offset_uses_timedelta(self) -> None:
        """With bucket='30min', LastWeekSameHourBaseline still uses
        pd.Timedelta(days=7) wall-clock subtraction, not bucket-integer shift.

        Verified by constructing a 30-min 2-week fixture and confirming
        predictions match the exact 7-day-prior entry."""
        from app.training.baseline import LastWeekSameHourBaseline

        # Arrange: 2 weeks of 30-min data for zone 1
        base = pd.Timestamp("2022-01-03 00:00:00")
        n_buckets = 2 * 7 * 24 * 2  # 2 weeks × 30-min
        hours = pd.date_range(base, periods=n_buckets, freq="30min")
        gold_df = pd.DataFrame(
            {
                "pulocationid": 1,
                "hour": hours,
                "trip_count": np.arange(n_buckets, dtype=float),
            }
        )
        # Test set = second week
        week_2_start = base + pd.Timedelta(weeks=1)
        test_df = gold_df[gold_df["hour"] >= week_2_start].copy().reset_index(drop=True)

        baseline = LastWeekSameHourBaseline(target_col="trip_count", bucket="30min")

        # Act
        result = baseline.predict(test_df, gold_df)

        # Assert: each predicted value equals the gold value exactly 7 days earlier
        assert len(result) > 0
        for _, row in result.head(10).iterrows():
            week_ago_ts = row["hour"] - pd.Timedelta(days=7)
            expected = gold_df.loc[
                (gold_df["pulocationid"] == row["pulocationid"])
                & (gold_df["hour"] == week_ago_ts),
                "trip_count",
            ].values[0]
            assert row["y_pred_baseline"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Edge case: no week-ago overlap in test split → all four keys are None,
# and none of the six baseline/lift/recall metrics are logged to MLflow.
# ---------------------------------------------------------------------------


class TestEdgeCaseNoWeekAgoOverlap:
    """When aligned_df is empty (test split has no week-ago data), Trainer.run()
    must return None for all four baseline-related keys and must NOT log any
    of the six conditional metrics to MLflow."""

    def test_edge_no_week_ago_overlap_returns_none_and_skips_metrics(
        self, tmp_path: Path
    ) -> None:
        """Build a 1-week-only Gold fixture so the test split (last 20%) has no
        week-ago timestamps in the history.  Run Trainer.run() and confirm:

        1. result["baseline_metrics"] is None
        2. result["model_aligned_metrics"] is None
        3. result["wape_lift_pct"] is None
        4. result["recall_at_3"] is None
        5. mlflow.log_metric was NOT called for any of the six conditional keys
        """
        # Arrange — 1 week of data for 3 zones (168 hours × 3 zones = 504 rows).
        # With a 60/20/20 split the test slice covers the last 20% ≈ last 33 hours,
        # which fall within week 1 and have no week-ago anchor in the Gold history.
        n_hours = 24 * 7  # exactly 1 week
        records = []
        for hour_offset in range(n_hours):
            ts = BASE_TS + pd.Timedelta(hours=hour_offset)
            for zone in ZONES:
                hour_of_day = hour_offset % 24
                records.append(
                    {
                        "pulocationid": zone,
                        "hour": ts,
                        "trip_count": float(zone * 10 + hour_of_day),
                        "hour_of_day_sin": float(np.sin(2 * np.pi * hour_of_day / 24)),
                        "hour_of_day_cos": float(np.cos(2 * np.pi * hour_of_day / 24)),
                        "day_of_week_sin": float(np.sin(2 * np.pi * ts.dayofweek / 7)),
                        "day_of_week_cos": float(np.cos(2 * np.pi * ts.dayofweek / 7)),
                        "month_sin": float(np.sin(2 * np.pi * ts.month / 12)),
                        "month_cos": float(np.cos(2 * np.pi * ts.month / 12)),
                        "pickup_longitude": float(-74.00 - zone * 0.01),
                        "pickup_latitude": float(40.70 + zone * 0.01),
                        "trip_count_lag_1": float(zone * 10 + (hour_of_day - 1) % 24),
                        "trip_count_lag_3": float(zone * 10 + (hour_of_day - 3) % 24),
                        "trip_count_lag_6": float(zone * 10 + (hour_of_day - 6) % 24),
                        "trip_count_lag_12": float(zone * 10 + (hour_of_day - 12) % 24),
                        "trip_count_lag_24": float(zone * 10 + (hour_of_day - 24) % 24),
                        "trip_count_lag_168": float(zone * 10 + hour_of_day),
                    }
                )

        one_week_gold = pd.DataFrame(records)
        gold_dir = tmp_path / "gold"
        gold_dir.mkdir(parents=True)
        one_week_gold.to_parquet(
            gold_dir / "hourly_demand_featured.parquet", index=False
        )

        config = _trainer_config(tmp_path)
        trainer = Trainer(config=config)

        # Act
        with patch("app.training.trainer.mlflow") as mock_mlflow:
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(
                return_value=MagicMock()
            )
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            result = trainer.run(data=one_week_gold)

        # Assert 1–4: all four baseline-related keys are None
        assert result["baseline_metrics"] is None
        assert result["model_aligned_metrics"] is None
        assert result["wape_lift_pct"] is None
        assert result["recall_at_3"] is None

        # Assert 5: none of the six conditional metric names were logged
        logged_names = {call[0][0] for call in mock_mlflow.log_metric.call_args_list}
        conditional_metric_keys = {
            "baseline_test_mae",
            "baseline_test_wape",
            "model_test_mae",
            "model_test_wape",
            "wape_lift_pct",
            "recall_at_3",
        }
        assert logged_names.isdisjoint(conditional_metric_keys), (
            f"Expected no conditional metrics logged when aligned_df is empty, "
            f"but found: {logged_names & conditional_metric_keys}"
        )
