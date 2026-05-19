"""Tests for the inference module (app/inference/predictor.py)."""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

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

    def test_finite_values(
        self, predictor: Predictor, gold_df: pd.DataFrame
    ) -> None:
        """All predictions are finite numbers."""
        predictor.load_model()
        preds = predictor.predict(gold_df)
        assert np.all(np.isfinite(preds))


# -- Test: build_predictions_table ------------------------------------------


class TestBuildPredictionsTable:
    """Tests for Predictor.build_predictions_table."""

    def test_correct_columns(
        self, predictor: Predictor, gold_df: pd.DataFrame
    ) -> None:
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

    def test_metric_keys(
        self, predictor: Predictor, gold_df: pd.DataFrame
    ) -> None:
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

    def test_file_created(
        self, predictor: Predictor, gold_df: pd.DataFrame
    ) -> None:
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
