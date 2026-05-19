"""Tests for the training module (app/training/trainer.py)."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest

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


@pytest.fixture
def trainer() -> Trainer:
    """Return a Trainer with minimal test config."""
    return Trainer(config=MINIMAL_CONFIG)


@pytest.fixture
def gold_df() -> pd.DataFrame:
    """Return a synthetic Gold layer DataFrame."""
    return _make_gold_df()


# -- Test: split_data -------------------------------------------------------


class TestSplitData:
    """Tests for Trainer.split_data."""

    def test_split_ratios(
        self, trainer: Trainer, gold_df: pd.DataFrame
    ) -> None:
        """Train/val/test sets have approximately correct proportions."""
        train, val, test = trainer.split_data(gold_df)

        n = len(gold_df)
        assert len(train) + len(val) + len(test) == n
        assert abs(len(train) / n - 0.8) < 0.02
        assert abs(len(val) / n - 0.1) < 0.02
        assert abs(len(test) / n - 0.1) < 0.02

    def test_all_splits_non_empty(
        self, trainer: Trainer, gold_df: pd.DataFrame
    ) -> None:
        """All three splits contain at least one row."""
        train, val, test = trainer.split_data(gold_df)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_no_temporal_leakage(
        self, trainer: Trainer, gold_df: pd.DataFrame
    ) -> None:
        """Max hour in train < min hour in val < min hour in test."""
        train, val, test = trainer.split_data(gold_df)

        assert train["hour"].max() < val["hour"].min()
        assert val["hour"].max() < test["hour"].min()

    def test_no_overlap(
        self, trainer: Trainer, gold_df: pd.DataFrame
    ) -> None:
        """No overlapping timestamps between splits."""
        train, val, test = trainer.split_data(gold_df)

        train_hours = set(train["hour"])
        val_hours = set(val["hour"])
        test_hours = set(test["hour"])

        assert train_hours.isdisjoint(val_hours)
        assert val_hours.isdisjoint(test_hours)
        assert train_hours.isdisjoint(test_hours)

    def test_deterministic(
        self, trainer: Trainer, gold_df: pd.DataFrame
    ) -> None:
        """Calling split_data twice produces identical results."""
        train1, val1, test1 = trainer.split_data(gold_df)
        train2, val2, test2 = trainer.split_data(gold_df)

        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)
        pd.testing.assert_frame_equal(test1, test2)

    def test_sorted_within_splits(
        self, trainer: Trainer, gold_df: pd.DataFrame
    ) -> None:
        """Each split is sorted by hour."""
        train, val, test = trainer.split_data(gold_df)

        assert train["hour"].is_monotonic_increasing
        assert val["hour"].is_monotonic_increasing
        assert test["hour"].is_monotonic_increasing


# -- Test: train -------------------------------------------------------------


class TestTrainModel:
    """Tests for Trainer.train."""

    def test_model_is_fitted(
        self, trainer: Trainer, gold_df: pd.DataFrame
    ) -> None:
        """Model has fitted attributes after training."""
        train_df, _, _ = trainer.split_data(gold_df)
        x_train = train_df[trainer.features]
        y_train = train_df[trainer.target]

        trainer.train(x_train, y_train)

        assert hasattr(trainer.model, "n_features_in_")
        assert trainer.model.n_features_in_ == len(trainer.features)

    def test_model_can_predict(
        self, trainer: Trainer, gold_df: pd.DataFrame
    ) -> None:
        """Fitted model produces predictions."""
        train_df, val_df, _ = trainer.split_data(gold_df)
        trainer.train(train_df[trainer.features], train_df[trainer.target])

        preds = trainer.model.predict(val_df[trainer.features])
        assert len(preds) == len(val_df)


# -- Test: evaluate ----------------------------------------------------------


class TestEvaluate:
    """Tests for Trainer.evaluate."""

    def test_metric_keys(
        self, trainer: Trainer, gold_df: pd.DataFrame
    ) -> None:
        """Returns dict with expected metric keys."""
        train_df, val_df, _ = trainer.split_data(gold_df)
        trainer.train(train_df[trainer.features], train_df[trainer.target])
        preds = trainer.model.predict(val_df[trainer.features])

        metrics = trainer.evaluate(val_df[trainer.target], preds)

        assert set(metrics.keys()) == {"mae", "rmse", "mape", "wape"}

    def test_metrics_non_negative(
        self, trainer: Trainer, gold_df: pd.DataFrame
    ) -> None:
        """All metric values are non-negative."""
        train_df, val_df, _ = trainer.split_data(gold_df)
        trainer.train(train_df[trainer.features], train_df[trainer.target])
        preds = trainer.model.predict(val_df[trainer.features])

        metrics = trainer.evaluate(val_df[trainer.target], preds)

        for key, value in metrics.items():
            assert value >= 0, f"{key} is negative: {value}"

    def test_mae_leq_rmse(
        self, trainer: Trainer, gold_df: pd.DataFrame
    ) -> None:
        """MAE <= RMSE by definition."""
        train_df, val_df, _ = trainer.split_data(gold_df)
        trainer.train(train_df[trainer.features], train_df[trainer.target])
        preds = trainer.model.predict(val_df[trainer.features])

        metrics = trainer.evaluate(val_df[trainer.target], preds)

        assert metrics["mae"] <= metrics["rmse"]


# -- Test: run (save model) --------------------------------------------------


class TestRunSavesModel:
    """Tests for Trainer.run — model artifact saving."""

    def test_model_file_created(
        self, trainer: Trainer, gold_df: pd.DataFrame, tmp_path: Any
    ) -> None:
        """run() saves a model.pkl file that can be reloaded."""
        trainer.store_path = str(tmp_path / "models")
        trainer.tracking_uri = str(tmp_path / "mlruns")

        result = trainer.run(data=gold_df)

        model_path = result["model_path"]
        assert os.path.exists(model_path)

        loaded = joblib.load(model_path)
        preds = loaded.predict(gold_df[trainer.features].head(5))
        assert len(preds) == 5


# -- Test: run (MLflow logging) ----------------------------------------------


class TestRunLogsToMlflow:
    """Tests for Trainer.run — MLflow integration."""

    def test_mlflow_run_created(
        self, trainer: Trainer, gold_df: pd.DataFrame, tmp_path: Any
    ) -> None:
        """run() creates an MLflow run with params and metrics."""
        trainer.store_path = str(tmp_path / "models")
        trainer.tracking_uri = str(tmp_path / "mlruns")

        with patch("app.training.trainer.mlflow") as mock_mlflow:
            mock_run = mock_mlflow.start_run.return_value.__enter__.return_value
            trainer.run(data=gold_df)

            mock_mlflow.set_tracking_uri.assert_called_once_with(
                trainer.tracking_uri
            )
            mock_mlflow.set_experiment.assert_called_once_with(
                trainer.experiment_name
            )
            mock_mlflow.start_run.assert_called_once()

            # Params logged
            mock_mlflow.log_params.assert_called_once_with(
                trainer.model_params
            )

            # Metrics logged (val + test × 4 metrics each)
            metric_calls = mock_mlflow.log_metric.call_args_list
            metric_names = {call[0][0] for call in metric_calls}
            for prefix in ["val_", "test_"]:
                for key in ["mae", "rmse", "mape", "wape"]:
                    assert f"{prefix}{key}" in metric_names

            # Model logged
            mock_mlflow.lightgbm.log_model.assert_called_once()
