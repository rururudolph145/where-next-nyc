"""Tests for the inference module (app/inference/predictor.py and simulator.py)."""

from __future__ import annotations

import logging
import os
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.inference.exceptions import NoApprovedModelError
from app.inference.predictor import Predictor
from app.training.trainer import Trainer
from tests.conftest import make_mock_version


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

MINIMAL_CONFIG: dict[str, Any] = {
    "env": "local",
    "data": {"gold_path": "data/gold/yellow_tripdata"},
    "train": {
        "features": FEATURES,
        "target": "trip_count",
        "split_ratios": [0.8, 0.1, 0.1],
    },
    "model": {
        "name": "LGBMRegressor",
        "params": {
            "n_estimators": 10,
            "learning_rate": 0.1,
            "num_leaves": 8,
            "random_state": 42,
            "verbosity": -1,
        },
        "store_path": "models/",
    },
    "inference": {
        "output_path": "data/predictions/demand_predictions.parquet",
    },
    "mlflow": {
        "experiment_name": "test_experiment",
        "tracking_uri": "mlruns",
    },
}


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
    config = {**MINIMAL_CONFIG}
    config = {
        **config,
        "model": {
            **config["model"],
            "store_path": str(tmp_path / "models"),
        },
        "mlflow": {
            **config["mlflow"],
            "tracking_uri": str(tmp_path / "mlruns"),
        },
    }

    trainer = Trainer(config)
    trainer.run(data=gold_df)
    return str(tmp_path / "models")


@pytest.fixture()
def predictor(trained_model_path: str, tmp_path: Any) -> Predictor:
    """Return a Predictor with a trained model available."""
    config = {
        **MINIMAL_CONFIG,
        "model": {**MINIMAL_CONFIG["model"], "store_path": trained_model_path},
        "inference": {
            "output_path": str(tmp_path / "predictions" / "demand_predictions.parquet"),
        },
    }
    return Predictor(config)


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
        config = {
            **MINIMAL_CONFIG,
            "model": {
                **MINIMAL_CONFIG["model"],
                "store_path": str(tmp_path / "nonexistent"),
            },
        }
        pred = Predictor(config)
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

    # NOTE: This test exercises Predictor.build_predictions_table, which is
    # out of scope for PBI 21. The simulator's parquet (which DOES include
    # model_version + served_from_stage) is covered by
    # TestFallback::test_simulator_parquet_columns_exact_set below.
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


# ---------------------------------------------------------------------------
# Helpers for HourlySimulator / fallback tests
# ---------------------------------------------------------------------------

SIM_FEATURES = [
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

SIMULATOR_CONFIG: dict[str, Any] = {
    "data": {"gold_path": "data/gold/yellow_tripdata"},
    "train": {
        "features": SIM_FEATURES,
        "target": "trip_count",
    },
    "model": {"store_path": "models/", "name": "LGBMRegressor", "params": {}},
    "mlflow": {
        "tracking_uri": "mlruns",
        "model_name": "ny_taxi_demand_model",
    },
    "inference": {
        "simulation": {
            "start_hour": "2022-01-01 00:00:00",
            "end_hour": "2022-01-01 02:00:00",
        },
    },
}


def _make_sim_gold_df() -> pd.DataFrame:
    """Build a tiny synthetic Gold DataFrame covering the simulator window."""
    rng = np.random.RandomState(0)
    hours = pd.date_range("2022-01-01 00:00:00", periods=3, freq="h")
    n = len(hours) * 5  # 5 zones per hour
    data: dict[str, Any] = {
        "hour": np.tile(hours, 5),
        "pulocationid": np.repeat(np.arange(1, 6), 3),
    }
    for feat in SIM_FEATURES:
        data[feat] = rng.rand(n)
    data["trip_count"] = rng.randint(0, 50, n)
    return pd.DataFrame(data)


def _make_stub_model() -> MagicMock:
    """Return a model stub that returns a constant prediction array."""
    stub = MagicMock()
    stub.predict.side_effect = lambda x: np.ones(len(x))
    return stub


# ---------------------------------------------------------------------------
# PBI 21 — TestFallback
# ---------------------------------------------------------------------------


class TestFallback:
    """Tests for HourlySimulator registry-aware model loading (PBI 21)."""

    def _make_simulator(self, tmp_path: Any) -> Any:
        """Build a HourlySimulator with tmp_path output and inline gold data."""
        from app.inference.simulator import HourlySimulator

        cfg = {
            **SIMULATOR_CONFIG,
            "inference": {
                **SIMULATOR_CONFIG.get("inference", {}),
                "output_path": str(
                    tmp_path / "predictions" / "demand_predictions.parquet"
                ),
            },
        }
        sim = HourlySimulator(config=cfg)
        return sim

    def test_production_present_uses_production(self, tmp_path: Any) -> None:
        """Production version available → loads Production model; parquet stamped correctly.

        AC #1, #5: served_from_stage=="Production", model_version=="3".
        """
        # Arrange
        sim = self._make_simulator(tmp_path)
        gold_df = _make_sim_gold_df()
        stub_model = _make_stub_model()
        mock_version = make_mock_version(version="3", stage="Production")

        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = [mock_version]

        # Act
        with patch("app.inference.simulator.MlflowClient", return_value=mock_client):
            with patch(
                "app.inference.simulator.mlflow.lightgbm.load_model",
                return_value=stub_model,
            ):
                with patch.object(sim, "load_gold", return_value=gold_df):
                    result = sim.run()

        # Assert
        predictions_df: pd.DataFrame = result["predictions_df"]
        assert "served_from_stage" in predictions_df.columns
        assert "model_version" in predictions_df.columns
        assert (predictions_df["served_from_stage"] == "Production").all()
        assert (predictions_df["model_version"] == "3").all()

    def test_production_absent_archived_present_uses_archived_with_warning(
        self, tmp_path: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No Production version, one Archived → loads Archived; warning emitted.

        AC #2, #5, #6: served_from_stage=="Archived", model_version=="2",
        warning contains model_fallback_active=true.
        """
        # Arrange
        sim = self._make_simulator(tmp_path)
        gold_df = _make_sim_gold_df()
        stub_model = _make_stub_model()
        archived_version = make_mock_version(version="2", stage="Archived")

        def get_latest_versions_side_effect(
            name: str, stages: list[str]
        ) -> list[MagicMock]:
            if "Production" in stages:
                return []
            return [archived_version]

        mock_client = MagicMock()
        mock_client.get_latest_versions.side_effect = get_latest_versions_side_effect

        # Act
        with caplog.at_level(logging.WARNING, logger="app.inference.simulator"):
            with patch(
                "app.inference.simulator.MlflowClient", return_value=mock_client
            ):
                with patch(
                    "app.inference.simulator.mlflow.lightgbm.load_model",
                    return_value=stub_model,
                ):
                    with patch.object(sim, "load_gold", return_value=gold_df):
                        result = sim.run()

        # Assert parquet values
        predictions_df: pd.DataFrame = result["predictions_df"]
        assert (predictions_df["served_from_stage"] == "Archived").all()
        assert (predictions_df["model_version"] == "2").all()
        # Assert warning was emitted
        assert any(
            "model_fallback_active=true" in record.message for record in caplog.records
        )

    def test_both_absent_raises_no_approved_model_error(self, tmp_path: Any) -> None:
        """Neither Production nor Archived → raises NoApprovedModelError; no parquet written.

        AC #3, #11.
        """
        # Arrange
        sim = self._make_simulator(tmp_path)
        output_path = sim.output_path
        gold_df = _make_sim_gold_df()

        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []

        # Act + Assert
        with patch("app.inference.simulator.MlflowClient", return_value=mock_client):
            with patch.object(sim, "load_gold", return_value=gold_df):
                with pytest.raises(NoApprovedModelError):
                    sim.run()

        assert not os.path.exists(output_path)

    def test_artifact_download_failure_falls_back_to_archived(
        self, tmp_path: Any
    ) -> None:
        """Production artifact raises MlflowException → falls back to Archived model.

        AC #4.
        """
        from mlflow.exceptions import MlflowException

        # Arrange
        sim = self._make_simulator(tmp_path)
        gold_df = _make_sim_gold_df()
        stub_model = _make_stub_model()
        prod_version = make_mock_version(version="5", stage="Production")
        arch_version = make_mock_version(version="4", stage="Archived")

        def get_latest_versions_side_effect(
            name: str, stages: list[str]
        ) -> list[MagicMock]:
            if "Production" in stages:
                return [prod_version]
            return [arch_version]

        mock_client = MagicMock()
        mock_client.get_latest_versions.side_effect = get_latest_versions_side_effect

        call_count = {"n": 0}

        def load_model_side_effect(uri: str) -> MagicMock:
            call_count["n"] += 1
            if "5" in uri:
                raise MlflowException("artifact missing")
            return stub_model

        # Act
        with patch("app.inference.simulator.MlflowClient", return_value=mock_client):
            with patch(
                "app.inference.simulator.mlflow.lightgbm.load_model",
                side_effect=load_model_side_effect,
            ):
                with patch.object(sim, "load_gold", return_value=gold_df):
                    result = sim.run()

        # Assert Archived model was used
        predictions_df: pd.DataFrame = result["predictions_df"]
        assert (predictions_df["served_from_stage"] == "Archived").all()
        assert (predictions_df["model_version"] == "4").all()

    def test_mlflow_client_constructor_raises_falls_back_to_archived(
        self, tmp_path: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """MlflowClient constructor raises MlflowException → fallback to Archived succeeds.

        AC #4, edge case: missing mlflow.db on fresh checkout.
        """
        from mlflow.exceptions import MlflowException

        # Arrange
        sim = self._make_simulator(tmp_path)
        gold_df = _make_sim_gold_df()
        stub_model = _make_stub_model()
        arch_version = make_mock_version(version="1", stage="Archived")

        constructor_call = {"n": 0}

        def client_constructor_side_effect(**kwargs: Any) -> MagicMock:
            constructor_call["n"] += 1
            if constructor_call["n"] == 1:
                raise MlflowException("mlflow.db not found")
            # Second call (for Archived) succeeds
            mock_client = MagicMock()
            mock_client.get_latest_versions.return_value = [arch_version]
            return mock_client

        # Act
        with caplog.at_level(logging.WARNING, logger="app.inference.simulator"):
            with patch(
                "app.inference.simulator.MlflowClient",
                side_effect=client_constructor_side_effect,
            ):
                with patch(
                    "app.inference.simulator.mlflow.lightgbm.load_model",
                    return_value=stub_model,
                ):
                    with patch.object(sim, "load_gold", return_value=gold_df):
                        result = sim.run()

        # Assert fallback warning emitted and model loaded from Archived
        assert any(
            "model_fallback_active=true" in record.message for record in caplog.records
        )
        predictions_df: pd.DataFrame = result["predictions_df"]
        assert (predictions_df["served_from_stage"] == "Archived").all()

    def test_archived_list_empty_on_fresh_checkout_raises(self, tmp_path: Any) -> None:
        """Production [], Archived [] → NoApprovedModelError names both stages.

        AC #3, edge case: fresh checkout with no registry versions.
        """
        # Arrange
        sim = self._make_simulator(tmp_path)

        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []

        # Act + Assert
        with patch("app.inference.simulator.MlflowClient", return_value=mock_client):
            with pytest.raises(NoApprovedModelError) as exc_info:
                sim._load_model_from_registry()

        error_message = str(exc_info.value)
        assert "Production" in error_message
        assert "Archived" in error_message
        assert "ny_taxi_demand_model" in error_message

    def test_simulator_parquet_columns_exact_set(self, tmp_path: Any) -> None:
        """Simulator parquet column set exactly matches the PBI-21 schema.

        AC-7 (corrected): the simulator's output parquet — not Predictor — must
        contain exactly the six expected columns.

        Arrange: Production version available; stub model; synthetic gold data.
        Act: HourlySimulator.run() (same mocking pattern as
             test_production_present_uses_production).
        Assert: set(parquet.columns) == {
                  "pulocationid", "hour", "predicted_trip_count",
                  "actual_trip_count", "model_version", "served_from_stage"
                }.
        """
        # Arrange
        sim = self._make_simulator(tmp_path)
        gold_df = _make_sim_gold_df()
        stub_model = _make_stub_model()
        mock_version = make_mock_version(version="3", stage="Production")

        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = [mock_version]

        # Act
        with patch("app.inference.simulator.MlflowClient", return_value=mock_client):
            with patch(
                "app.inference.simulator.mlflow.lightgbm.load_model",
                return_value=stub_model,
            ):
                with patch.object(sim, "load_gold", return_value=gold_df):
                    result = sim.run()

        # Assert — read back the written parquet to verify on-disk schema
        parquet = pd.read_parquet(result["output_path"])
        expected_columns = {
            "pulocationid",
            "hour",
            "predicted_trip_count",
            "actual_trip_count",
            "model_version",
            "served_from_stage",
        }
        assert set(parquet.columns) == expected_columns
