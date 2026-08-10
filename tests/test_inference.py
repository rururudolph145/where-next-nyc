"""Tests for the inference module (app/inference/predictor.py)."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.core.config import AppConfig
from app.inference.predictor import Predictor
from app.training.trainer import Trainer


# -- Shared config and data fixtures ----------------------------------------

FEATURES = [
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


def _predictor_config(
    *,
    store_path: str = "models/",
    output_path: str = "data/predictions/demand_predictions.parquet",
    tracking_uri: str = "mlruns",
) -> AppConfig:
    """Build an AppConfig for Predictor tests."""
    from tests.conftest import _FULL_CONFIG_DICT

    base = AppConfig.model_validate(_FULL_CONFIG_DICT)
    return base.model_copy(
        update={
            "train": base.train.model_copy(
                update={"features": FEATURES, "target": "trip_count"}
            ),
            "model": base.model.model_copy(
                update={
                    "params": {
                        "n_estimators": 10,
                        "learning_rate": 0.1,
                        "num_leaves": 8,
                        "random_state": 42,
                        "verbosity": -1,
                    },
                    "store_path": store_path,
                }
            ),
            "inference": base.inference.model_copy(update={"output_path": output_path}),
            "mlflow": base.mlflow.model_copy(
                update={
                    "experiment_name": "test_experiment",
                    "tracking_uri": tracking_uri,
                }
            ),
        }
    )


def _make_gold_df(n_rows: int = 1000) -> pd.DataFrame:
    """Create a synthetic Gold layer DataFrame for testing."""
    rng = np.random.RandomState(42)
    hours = pd.date_range("2022-01-01", periods=n_rows, freq="h")
    data: dict[str, Any] = {"hour": hours, "pulocationid": rng.randint(1, 50, n_rows)}

    for feat in FEATURES:
        data[feat] = rng.rand(n_rows)

    data["trip_count"] = rng.randint(0, 100, n_rows)
    return pd.DataFrame(data)


@pytest.fixture()
def gold_df() -> pd.DataFrame:
    """Return a synthetic Gold layer DataFrame."""
    return _make_gold_df()


@pytest.fixture()
def trained_model_path(tmp_path: Any, gold_df: pd.DataFrame) -> str:
    """Train a model and save it to tmp_path, returning the store path."""
    trainer_cfg = _predictor_config(
        store_path=str(tmp_path / "models"),
        tracking_uri=str(tmp_path / "mlruns"),
    )
    trainer = Trainer(trainer_cfg)
    trainer.run(data=gold_df)
    return str(tmp_path / "models")


@pytest.fixture()
def predictor(trained_model_path: str, tmp_path: Any) -> Predictor:
    """Return a Predictor with a trained model available."""
    return Predictor(
        _predictor_config(
            store_path=trained_model_path,
            output_path=str(tmp_path / "predictions" / "demand_predictions.parquet"),
            tracking_uri=str(tmp_path / "mlruns"),
        )
    )


# -- Test: load_model -------------------------------------------------------


class TestLoadModel:
    """Tests for Predictor.load_model."""

    def test_loads_saved_model(
        self, predictor: Predictor, trained_model_path: str
    ) -> None:
        """Model loads successfully from disk."""
        model = predictor.load_model()
        assert model is not None
        assert hasattr(model, "predict")

    def test_errors_on_missing_file(self, tmp_path: Any) -> None:
        """Raises FileNotFoundError when model file does not exist."""
        pred = Predictor(_predictor_config(store_path=str(tmp_path / "nonexistent")))
        with pytest.raises(FileNotFoundError):
            pred.load_model()


# -- Test: predict -----------------------------------------------------------


class TestPredict:
    """Tests for Predictor.predict."""

    def test_correct_output_shape(
        self, predictor: Predictor, gold_df: pd.DataFrame
    ) -> None:
        """Predictions have the same length as input."""
        predictor.load_model()
        preds = predictor.predict(gold_df)
        assert len(preds) == len(gold_df)

    def test_finite_values(self, predictor: Predictor, gold_df: pd.DataFrame) -> None:
        """All predictions are finite numbers."""
        predictor.load_model()
        preds = predictor.predict(gold_df)
        assert np.all(np.isfinite(preds))


# -- Test: build_predictions_table ------------------------------------------


class TestBuildPredictionsTable:
    """Tests for Predictor.build_predictions_table."""

    def test_correct_columns(self, predictor: Predictor, gold_df: pd.DataFrame) -> None:
        """Output has expected columns."""
        predictor.load_model()
        preds = predictor.predict(gold_df)
        table = predictor.build_predictions_table(gold_df, preds)

        expected_cols = {
            "pulocationid",
            "hour",
            "predicted_trip_count",
            "actual_trip_count",
        }
        assert set(table.columns) == expected_cols

    def test_row_count_matches(
        self, predictor: Predictor, gold_df: pd.DataFrame
    ) -> None:
        """Output row count matches input."""
        predictor.load_model()
        preds = predictor.predict(gold_df)
        table = predictor.build_predictions_table(gold_df, preds)
        assert len(table) == len(gold_df)

    def test_zone_ids_preserved(
        self, predictor: Predictor, gold_df: pd.DataFrame
    ) -> None:
        """Zone IDs in output match input."""
        predictor.load_model()
        preds = predictor.predict(gold_df)
        table = predictor.build_predictions_table(gold_df, preds)
        np.testing.assert_array_equal(
            table["pulocationid"].values, gold_df["pulocationid"].values
        )


# -- Test: evaluate ----------------------------------------------------------


class TestEvaluate:
    """Tests for Predictor.evaluate."""

    def test_metric_keys(self, predictor: Predictor, gold_df: pd.DataFrame) -> None:
        """Returns dict with expected metric keys."""
        predictor.load_model()
        preds = predictor.predict(gold_df)
        metrics = predictor.evaluate(gold_df["trip_count"], preds)
        assert set(metrics.keys()) == {"mae", "rmse", "mape", "wape"}

    def test_metrics_non_negative(
        self, predictor: Predictor, gold_df: pd.DataFrame
    ) -> None:
        """All metric values are non-negative."""
        predictor.load_model()
        preds = predictor.predict(gold_df)
        metrics = predictor.evaluate(gold_df["trip_count"], preds)
        for key, value in metrics.items():
            assert value >= 0, f"{key} is negative: {value}"


# -- Test: save_predictions -------------------------------------------------


class TestSavePredictions:
    """Tests for Predictor.save_predictions."""

    def test_file_created(self, predictor: Predictor, gold_df: pd.DataFrame) -> None:
        """Parquet file is created at configured path."""
        predictor.load_model()
        preds = predictor.predict(gold_df)
        table = predictor.build_predictions_table(gold_df, preds)
        path = predictor.save_predictions(table)
        assert os.path.exists(path)

    def test_roundtrip_readable(
        self, predictor: Predictor, gold_df: pd.DataFrame
    ) -> None:
        """Saved file can be read back and matches original."""
        predictor.load_model()
        preds = predictor.predict(gold_df)
        table = predictor.build_predictions_table(gold_df, preds)
        path = predictor.save_predictions(table)

        loaded = pd.read_parquet(path)
        assert len(loaded) == len(table)
        assert set(loaded.columns) == set(table.columns)

    def test_history_partition_written(
        self, predictor: Predictor, gold_df: pd.DataFrame
    ) -> None:
        """save_predictions writes a single flat history/dt=<date>.parquet (no hour subdirs)."""
        # Arrange
        predictor.load_model()
        preds = predictor.predict(gold_df)
        table = predictor.build_predictions_table(gold_df, preds)
        # restrict to a single date so the assertion is exact
        one_day = table[table["hour"].dt.date == table["hour"].dt.date.min()].copy()

        # Act
        predictor.save_predictions(one_day)

        # Assert — flat file, no per-hour subdirectory
        pred_dir = os.path.dirname(predictor.output_path)
        date_str = one_day["hour"].dt.date.min().isoformat()
        day_file = os.path.join(pred_dir, "history", f"dt={date_str}.parquet")
        assert os.path.isfile(day_file), f"day-partition file missing: {day_file}"
        assert not os.path.isdir(os.path.join(pred_dir, "history", f"dt={date_str}")), (
            "old per-hour dir should not exist"
        )
        # round-trip: day file holds exactly the input rows with the same columns
        loaded = pd.read_parquet(day_file)
        assert len(loaded) == len(one_day)
        assert set(loaded.columns) == set(one_day.columns)

    def test_history_multiday_partitions(
        self, predictor: Predictor, gold_df: pd.DataFrame
    ) -> None:
        """A batch spanning >=2 dates produces one dt=<date>.parquet file per date."""
        # Arrange
        predictor.load_model()
        preds = predictor.predict(gold_df)
        table = predictor.build_predictions_table(gold_df, preds)
        expected_dates = {d.isoformat() for d in table["hour"].dt.date.unique()}
        assert len(expected_dates) >= 2  # gold_df spans ~42 days

        # Act
        predictor.save_predictions(table)

        # Assert — flat .parquet files, not directories
        history_root = os.path.join(os.path.dirname(predictor.output_path), "history")
        got_files = {f for f in os.listdir(history_root) if f.endswith(".parquet")}
        expected_files = {f"dt={d}.parquet" for d in expected_dates}
        assert got_files == expected_files

    def test_history_write_failure_isolated(
        self, predictor: Predictor, gold_df: pd.DataFrame
    ) -> None:
        """A history-write failure never breaks the hot-slot nor propagates."""
        # Arrange
        predictor.load_model()
        preds = predictor.predict(gold_df)
        table = predictor.build_predictions_table(gold_df, preds)
        date_str = table["hour"].dt.date.min().isoformat()

        # Act — force the history write to fail; hot-slot write must still land
        real_makedirs = os.makedirs

        def makedirs_side_effect(path: str, **kwargs: Any) -> None:
            if "/history/" in str(path) or str(path).endswith("/history"):
                raise OSError("boom")
            real_makedirs(path, **kwargs)

        with patch(
            "app.inference.predictor.os.makedirs", side_effect=makedirs_side_effect
        ):
            path = predictor.save_predictions(table)

        # Assert
        assert os.path.exists(path)  # hot-slot written
        pred_dir = os.path.dirname(predictor.output_path)
        assert not os.path.isfile(
            os.path.join(pred_dir, "history", f"dt={date_str}.parquet")
        )

    def test_upsert_appends_across_ticks(
        self, predictor: Predictor, gold_df: pd.DataFrame
    ) -> None:
        """Two consecutive saves to the same date accumulate all buckets.

        Simulates two DAG ticks writing non-overlapping 15-min buckets on the
        same calendar date — all 4 buckets must survive in the day file.
        """
        # Arrange
        predictor.load_model()
        preds = predictor.predict(gold_df)
        table = predictor.build_predictions_table(gold_df, preds)
        date_min = table["hour"].dt.date.min()
        same_day = table[table["hour"].dt.date == date_min].copy()
        # split into two non-overlapping bucket halves by sorted hour
        sorted_hours = sorted(same_day["hour"].unique())
        half = len(sorted_hours) // 2
        tick1 = same_day[same_day["hour"].isin(sorted_hours[:half])].copy()
        tick2 = same_day[same_day["hour"].isin(sorted_hours[half:])].copy()
        assert len(tick1) > 0 and len(tick2) > 0

        # Act — two sequential saves
        predictor.save_predictions(tick1)
        predictor.save_predictions(tick2)

        # Assert — day file contains all buckets from both ticks
        pred_dir = os.path.dirname(predictor.output_path)
        day_file = os.path.join(
            pred_dir, "history", f"dt={date_min.isoformat()}.parquet"
        )
        loaded = pd.read_parquet(day_file)
        assert len(loaded) == len(tick1) + len(tick2)

    def test_upsert_dedup_keeps_last(
        self, predictor: Predictor, gold_df: pd.DataFrame
    ) -> None:
        """Re-scoring the same (pulocationid, hour) keeps the second (newer) value.

        Pins the concat([existing, new], keep='last') order — reversing it would
        silently keep the stale value and this test would catch the regression.
        """
        # Arrange — build a two-row table with identical keys but different predictions
        predictor.load_model()
        fixed_hour = pd.Timestamp("2022-01-01 13:00:00")
        row_first = pd.DataFrame(
            {
                "pulocationid": [1],
                "hour": [fixed_hour],
                "predicted_trip_count": [10.0],
                "actual_trip_count": [5],
            }
        )
        row_second = pd.DataFrame(
            {
                "pulocationid": [1, 2],
                "hour": [fixed_hour, fixed_hour],
                "predicted_trip_count": [99.0, 7.0],  # zone 1 updated; zone 2 new
                "actual_trip_count": [5, 3],
            }
        )

        # Act — first tick, then second tick with updated zone-1 value
        predictor.save_predictions(row_first)
        predictor.save_predictions(row_second)

        # Assert — zone 1 holds the SECOND value (99.0), not the first (10.0)
        pred_dir = os.path.dirname(predictor.output_path)
        day_file = os.path.join(
            pred_dir, "history", f"dt={fixed_hour.date().isoformat()}.parquet"
        )
        loaded = pd.read_parquet(day_file)
        assert len(loaded) == 2  # zone 1 deduped + zone 2 new = 2 rows
        zone1_val = loaded.loc[
            loaded["pulocationid"] == 1, "predicted_trip_count"
        ].iloc[0]
        assert zone1_val == pytest.approx(99.0), (
            f"Expected 99.0 (second write) but got {zone1_val} — concat order is wrong"
        )


# -- Test: run (end-to-end) -------------------------------------------------


class TestRun:
    """Tests for Predictor.run — end-to-end orchestration."""

    def test_returns_expected_keys(
        self, predictor: Predictor, gold_df: pd.DataFrame
    ) -> None:
        """run() returns dict with metrics, output_path, n_predictions."""
        result = predictor.run(data=gold_df)
        assert "metrics" in result
        assert "output_path" in result
        assert "n_predictions" in result

    def test_predictions_file_exists(
        self, predictor: Predictor, gold_df: pd.DataFrame
    ) -> None:
        """Predictions file exists after run()."""
        result = predictor.run(data=gold_df)
        assert os.path.exists(result["output_path"])

    def test_n_predictions_matches(
        self, predictor: Predictor, gold_df: pd.DataFrame
    ) -> None:
        """n_predictions matches input row count."""
        result = predictor.run(data=gold_df)
        assert result["n_predictions"] == len(gold_df)

    def test_run_with_since_filters_gold(
        self, predictor: Predictor, gold_df: pd.DataFrame
    ) -> None:
        """run(data=gold, since=ts) scores only rows with hour >= since.

        n_predictions and the hot-slot row count must match the filtered slice.
        """
        # Arrange — pick the second distinct bucket as the since cutoff
        sorted_hours = sorted(gold_df["hour"].unique())
        since = sorted_hours[1]  # skip the very first bucket
        expected_rows = int((gold_df["hour"] >= since).sum())

        # Act
        result = predictor.run(data=gold_df, since=since)

        # Assert — n_predictions reflects the filtered slice
        assert result["n_predictions"] == expected_rows
        hot_slot = pd.read_parquet(result["output_path"])
        assert len(hot_slot) == expected_rows

    def test_run_without_since_unchanged(
        self, predictor: Predictor, gold_df: pd.DataFrame
    ) -> None:
        """run(data=gold) with no since argument scores all rows (regression pin)."""
        # Arrange / Act
        result = predictor.run(data=gold_df)

        # Assert — all input rows scored
        assert result["n_predictions"] == len(gold_df)

    def test_run_empty_slice_preserves_hot_slot(
        self, predictor: Predictor, gold_df: pd.DataFrame, tmp_path: Any
    ) -> None:
        """When since filters out ALL rows the hot-slot is left untouched.

        Overwriting the hot-slot with 0 rows would break /predict — this test
        guards Decision 3 from the plan.
        """
        # Arrange — seed hot-slot with a sentinel row
        sentinel = pd.DataFrame(
            {
                "pulocationid": [999],
                "hour": [pd.Timestamp("2000-01-01")],
                "predicted_trip_count": [42.0],
                "actual_trip_count": [42],
            }
        )
        os.makedirs(os.path.dirname(predictor.output_path), exist_ok=True)
        sentinel.to_parquet(predictor.output_path, index=False)

        # since is beyond every row in gold_df → empty slice
        beyond = gold_df["hour"].max() + pd.Timedelta(hours=1)

        # Act
        result = predictor.run(data=gold_df, since=beyond)

        # Assert — no predictions scored, hot-slot sentinel row survives
        assert result["n_predictions"] == 0
        hot_slot = pd.read_parquet(predictor.output_path)
        assert len(hot_slot) == 1
        assert hot_slot["predicted_trip_count"].iloc[0] == pytest.approx(42.0)


# -- Test: target decoupling (inference path) --------------------------------


class TestTargetDecoupling:
    """Tests for Predictor target-optional contract (inference decoupling)."""

    def test_build_predictions_table_nan_actual_when_target_absent(
        self, predictor: Predictor, gold_df: pd.DataFrame
    ) -> None:
        """build_predictions_table writes NaN actuals when target column absent.

        When gold has no trip_count column (inference path), actual_trip_count
        must be all-NaN — not a KeyError.
        """
        # Arrange
        predictor.load_model()
        gold_no_target = gold_df.drop(columns=["trip_count"])
        preds = predictor.predict(gold_no_target)

        # Act
        table = predictor.build_predictions_table(gold_no_target, preds)

        # Assert — actual_trip_count all-NaN, no KeyError, other cols intact
        assert table["actual_trip_count"].isna().all()
        assert "predicted_trip_count" in table.columns
        assert "pulocationid" in table.columns
        assert "hour" in table.columns

    def test_run_predicts_when_target_absent(
        self, predictor: Predictor, gold_df: pd.DataFrame
    ) -> None:
        """run(data=gold_no_target) succeeds with n_predictions > 0 and metrics == {}.

        The live DAG tick arrives with inference gold (no target column).
        The predictor must not KeyError and must return an empty metrics dict.
        """
        # Arrange
        gold_no_target = gold_df.drop(columns=["trip_count"])

        # Act
        result = predictor.run(data=gold_no_target)

        # Assert
        assert result["n_predictions"] > 0
        assert result["metrics"] == {}

    def test_run_evaluates_only_scorable_rows(
        self, predictor: Predictor, gold_df: pd.DataFrame
    ) -> None:
        """run() computes metrics on only the non-NaN subset of trip_count.

        When half the rows have NaN trip_count (partially-scored frame),
        metrics must be non-empty and contain the expected keys.
        """
        # Arrange — set half the trip_count values to NaN
        mixed = gold_df.copy()
        half = len(mixed) // 2
        mixed.loc[mixed.index[:half], "trip_count"] = np.nan

        # Act
        result = predictor.run(data=mixed)

        # Assert — metrics computed on the non-NaN half only
        assert result["metrics"] != {}
        assert set(result["metrics"].keys()) == {"mae", "rmse", "mape", "wape"}

    def test_run_with_target_present_metrics_nonempty(
        self, predictor: Predictor, gold_df: pd.DataFrame
    ) -> None:
        """Regression: run(data=gold_df) with full target yields non-empty metrics.

        Manual/batch runs with a known target must still score inline.
        Pins that the target-absent path doesn't break the normal path.
        """
        # Arrange / Act
        result = predictor.run(data=gold_df)

        # Assert
        assert result["metrics"] != {}
        assert set(result["metrics"].keys()) == {"mae", "rmse", "mape", "wape"}

    def test_history_actual_nan_survives_then_upsert_fills(
        self, predictor: Predictor, gold_df: pd.DataFrame
    ) -> None:
        """NaN actual_trip_count in history is overwritten by a later upsert.

        First run: inference gold (no target) → history rows with NaN actual.
        Second run: same (pulocationid, hour) with a known actual → upsert
        replaces NaN (keep='last' semantics), confirming NaN is not sticky.
        """
        # Arrange — inference run writes NaN actuals
        gold_no_target = gold_df.drop(columns=["trip_count"])
        predictor.run(data=gold_no_target)

        # Verify first run wrote NaN actuals in history
        history_dir = os.path.join(os.path.dirname(predictor.output_path), "history")
        assert os.path.exists(history_dir)
        first_parquets = [f for f in os.listdir(history_dir) if f.endswith(".parquet")]
        assert len(first_parquets) > 0
        first_history = pd.concat(
            [pd.read_parquet(os.path.join(history_dir, f)) for f in first_parquets]
        )
        assert first_history["actual_trip_count"].isna().all()

        # Act — second run with the same (pulocationid, hour) but known actuals
        predictor.run(data=gold_df)

        # Assert — after upsert, NaN is replaced by real actuals
        second_history = pd.concat(
            [
                pd.read_parquet(os.path.join(history_dir, f))
                for f in os.listdir(history_dir)
                if f.endswith(".parquet")
            ]
        )
        # Every (pulocationid, hour) pair kept only the latest write (keep='last')
        assert not second_history["actual_trip_count"].isna().all(), (
            "After upsert with known actuals, history should not be all-NaN"
        )


# ---------------------------------------------------------------------------
# PBI 37 Step 5 — Predictor registry-primary load
# ---------------------------------------------------------------------------


def _make_stub_model() -> MagicMock:
    """Return a model stub that returns a constant prediction array."""
    stub = MagicMock()
    stub.predict.side_effect = lambda x: np.ones(len(x))
    return stub


class TestPredictorRegistryLoad:
    """PBI 37 Step 5 — Predictor.load_model() consults MLflow @champion first."""

    def _make_predictor(self, tmp_path: Any) -> Predictor:
        return Predictor(
            _predictor_config(
                store_path=str(tmp_path / "models"),
                output_path=str(
                    tmp_path / "predictions" / "demand_predictions.parquet"
                ),
            )
        )

    def test_registry_primary_loads_via_champion_alias(self, tmp_path: Any) -> None:
        """mlflow.sklearn.load_model called with @champion URI; disk not read (PBI 37 Step 5 RED)."""
        # Arrange
        pred = self._make_predictor(tmp_path)
        stub_model = _make_stub_model()

        # Act
        with patch("app.inference.predictor.mlflow") as mock_mlflow:
            mock_mlflow.sklearn.load_model.return_value = stub_model
            result = pred.load_model()

        # Assert: registry was consulted; disk was NOT read
        mock_mlflow.sklearn.load_model.assert_called_once_with(
            "models:/ny_taxi_demand_model@champion"
        )
        assert result is stub_model

    def test_disk_fallback_on_mlflow_exception(self, tmp_path: Any) -> None:
        """mlflow raises MlflowException → falls back to joblib.load disk (PBI 37 Step 5 RED)."""
        from mlflow.exceptions import MlflowException
        from sklearn.dummy import DummyRegressor  # type: ignore[import-untyped]
        import joblib  # type: ignore[import-untyped]

        # Arrange: write a real (picklable) model to disk
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        dummy = DummyRegressor()
        dummy.fit([[0]], [0])
        joblib.dump(dummy, models_dir / "model.pkl")

        pred = self._make_predictor(tmp_path)

        # Act: registry raises → fallback to disk
        with patch("app.inference.predictor.mlflow") as mock_mlflow:
            mock_mlflow.sklearn.load_model.side_effect = MlflowException("no alias")
            result = pred.load_model()

        # Assert: disk model loaded
        assert result is not None
        assert hasattr(result, "predict")

    def test_both_absent_raises_file_not_found(self, tmp_path: Any) -> None:
        """Registry MlflowException AND no model.pkl → FileNotFoundError (PBI 37 Step 5 RED)."""
        from mlflow.exceptions import MlflowException

        # Arrange: no disk model
        pred = self._make_predictor(tmp_path)

        # Act + Assert
        with patch("app.inference.predictor.mlflow") as mock_mlflow:
            mock_mlflow.sklearn.load_model.side_effect = MlflowException("no alias")
            with pytest.raises(FileNotFoundError):
                pred.load_model()
