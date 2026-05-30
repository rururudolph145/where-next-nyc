"""Tests for the training module (app/training/trainer.py)."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytest

from app.core.metrics import evaluate as core_evaluate
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
    "feature": {"bucket": "h"},
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

# Config with registry + gate keys (PBI 20)
REGISTRY_CONFIG: dict[str, Any] = {
    "env": "local",
    "data": {"gold_path": "data/gold/yellow_tripdata"},
    "feature": {"bucket": "h"},
    "train": {
        "features": FEATURES,
        "target": "trip_count",
        "split_ratios": [0.8, 0.1, 0.1],
        "gate": {"min_wape_lift_pct": 2.0},
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
        "tracking_uri": "sqlite:///mlflow.db",
        "model_name": "ny_taxi_demand_model",
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

    def test_split_ratios(self, trainer: Trainer, gold_df: pd.DataFrame) -> None:
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

    def test_no_temporal_leakage(self, trainer: Trainer, gold_df: pd.DataFrame) -> None:
        """Max hour in train < min hour in val < min hour in test."""
        train, val, test = trainer.split_data(gold_df)

        assert train["hour"].max() < val["hour"].min()
        assert val["hour"].max() < test["hour"].min()

    def test_no_overlap(self, trainer: Trainer, gold_df: pd.DataFrame) -> None:
        """No overlapping timestamps between splits."""
        train, val, test = trainer.split_data(gold_df)

        train_hours = set(train["hour"])
        val_hours = set(val["hour"])
        test_hours = set(test["hour"])

        assert train_hours.isdisjoint(val_hours)
        assert val_hours.isdisjoint(test_hours)
        assert train_hours.isdisjoint(test_hours)

    def test_deterministic(self, trainer: Trainer, gold_df: pd.DataFrame) -> None:
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

    def test_model_is_fitted(self, trainer: Trainer, gold_df: pd.DataFrame) -> None:
        """Model has fitted attributes after training."""
        train_df, _, _ = trainer.split_data(gold_df)
        x_train = train_df[trainer.features]
        y_train = train_df[trainer.target]

        trainer.train(x_train, y_train)

        assert hasattr(trainer.model, "n_features_in_")
        assert trainer.model.n_features_in_ == len(trainer.features)

    def test_model_can_predict(self, trainer: Trainer, gold_df: pd.DataFrame) -> None:
        """Fitted model produces predictions."""
        train_df, val_df, _ = trainer.split_data(gold_df)
        trainer.train(train_df[trainer.features], train_df[trainer.target])

        preds = trainer.model.predict(val_df[trainer.features])
        assert len(preds) == len(val_df)


# -- Test: evaluate ----------------------------------------------------------


class TestEvaluate:
    """Tests for Trainer.evaluate."""

    def test_metric_keys(self, trainer: Trainer, gold_df: pd.DataFrame) -> None:
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

    def test_mae_leq_rmse(self, trainer: Trainer, gold_df: pd.DataFrame) -> None:
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
        """run() saves a model.pkl file that can be reloaded when gate promotes."""
        trainer.store_path = str(tmp_path / "models")
        trainer.tracking_uri = str(tmp_path / "mlruns")

        # Patch mlflow to avoid real SQLite I/O; gate auto-promotes (no incumbent)
        mock_mlflow = _make_mock_mlflow(registered_version="1", production_versions=[])

        # Override save_model to use real disk I/O so we can assert the file
        with patch("app.training.trainer.mlflow", mock_mlflow):
            result = trainer.run(data=gold_df)

        model_path = result["model_path"]
        assert model_path is not None
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

        # Use a properly configured mock to avoid TypeError inside apply_gate.
        mock_mlflow = _make_mock_mlflow(
            run_id="test_run", registered_version="1", production_versions=[]
        )

        with patch("app.training.trainer.mlflow", mock_mlflow):
            trainer.run(data=gold_df)

        # set_tracking_uri is called in __init__, not run(); see AC-4.
        mock_mlflow.set_experiment.assert_called_once_with(trainer.experiment_name)
        mock_mlflow.start_run.assert_called_once()

        # Params logged
        mock_mlflow.log_params.assert_called_once_with(trainer.model_params)

        # Metrics logged (val + test × 4 metrics each)
        metric_calls = mock_mlflow.log_metric.call_args_list
        metric_names = {c[0][0] for c in metric_calls}
        for prefix in ["val_", "test_"]:
            for key in ["mae", "rmse", "mape", "wape"]:
                assert f"{prefix}{key}" in metric_names

        # Model logged
        mock_mlflow.lightgbm.log_model.assert_called_once()

    def test_run_logs_baseline_and_lift_metrics(
        self, trainer: Trainer, gold_df: pd.DataFrame, tmp_path: Any
    ) -> None:
        """AC #1: all six new metric names are logged within the MLflow run."""
        trainer.store_path = str(tmp_path / "models")
        trainer.tracking_uri = str(tmp_path / "mlruns")

        mock_mlflow = _make_mock_mlflow(
            run_id="test_run", registered_version="1", production_versions=[]
        )

        with patch("app.training.trainer.mlflow", mock_mlflow):
            trainer.run(data=gold_df)

        metric_calls = mock_mlflow.log_metric.call_args_list
        metric_names = {c[0][0] for c in metric_calls}

        expected = {
            "baseline_test_mae",
            "baseline_test_wape",
            "model_test_mae",
            "model_test_wape",
            "wape_lift_pct",
            "recall_at_3",
        }
        assert expected.issubset(metric_names)

    def test_run_all_new_metrics_in_same_mlflow_run(
        self, trainer: Trainer, gold_df: pd.DataFrame, tmp_path: Any
    ) -> None:
        """AC #1: start_run called exactly once; all six metrics in that run."""
        trainer.store_path = str(tmp_path / "models")
        trainer.tracking_uri = str(tmp_path / "mlruns")

        mock_mlflow = _make_mock_mlflow(
            run_id="test_run", registered_version="1", production_versions=[]
        )

        with patch("app.training.trainer.mlflow", mock_mlflow):
            trainer.run(data=gold_df)

        mock_mlflow.start_run.assert_called_once()

        metric_calls = mock_mlflow.log_metric.call_args_list
        metric_names = {c[0][0] for c in metric_calls}
        for name in [
            "baseline_test_mae",
            "baseline_test_wape",
            "model_test_mae",
            "model_test_wape",
            "wape_lift_pct",
            "recall_at_3",
        ]:
            assert name in metric_names


# -- Test: registry gate (PBI 20) -------------------------------------------


def _make_mock_mlflow(
    run_id: str = "test_run_id",
    registered_version: str = "1",
    production_versions: list[Any] | None = None,
) -> MagicMock:
    """Build a pre-configured mlflow MagicMock for registry gate tests.

    Sets up:
    - start_run context manager returning an active_run with info.run_id
    - register_model returning a model version object with .version
    - tracking.MlflowClient() returning a mock client with configurable
      get_latest_versions result.
    """
    mock_mlflow = MagicMock()

    # Context manager: with mlflow.start_run() as active_run
    active_run = MagicMock()
    active_run.info.run_id = run_id
    mock_mlflow.start_run.return_value.__enter__.return_value = active_run

    # register_model → ModelVersion-like object
    mock_version = MagicMock()
    mock_version.version = registered_version
    mock_mlflow.register_model.return_value = mock_version

    # MlflowClient instance
    mock_client = MagicMock()
    mock_client.get_latest_versions.return_value = (
        production_versions if production_versions is not None else []
    )
    mock_mlflow.tracking.MlflowClient.return_value = mock_client

    return mock_mlflow


def _make_champion(version: str = "0", wape_tag: str | None = "15.0") -> MagicMock:
    """Return a ModelVersion mock representing the current Production champion."""
    champion = MagicMock()
    champion.version = version
    champion.tags = {"model_test_wape": wape_tag} if wape_tag is not None else {}
    return champion


class TestRegistryGate:
    """Tests for PBI 20: MLflow Model Registry + champion-vs-challenger gate."""

    # ------------------------------------------------------------------ #
    # Happy path                                                           #
    # ------------------------------------------------------------------ #

    def test_run_registers_model_version_when_run_succeeds(
        self, gold_df: pd.DataFrame, tmp_path: Any
    ) -> None:
        """mlflow.register_model called with correct run URI and model_name (AC-6, AC-7)."""
        # Arrange
        trainer = Trainer(config=REGISTRY_CONFIG)
        trainer.store_path = str(tmp_path / "models")
        mock_mlflow = _make_mock_mlflow(run_id="abc123", registered_version="1")

        # Act
        with patch("app.training.trainer.mlflow", mock_mlflow):
            trainer.run(data=gold_df)

        # Assert
        mock_mlflow.register_model.assert_called_once()
        args, _ = mock_mlflow.register_model.call_args
        assert args[0] == "runs:/abc123/model"
        assert args[1] == "ny_taxi_demand_model"

    def test_gate_promotes_when_no_production_version_exists(
        self, gold_df: pd.DataFrame, tmp_path: Any
    ) -> None:
        """transition_model_version_stage called with 'Production'; save_model called (AC-8)."""
        # Arrange
        trainer = Trainer(config=REGISTRY_CONFIG)
        trainer.store_path = str(tmp_path / "models")
        mock_mlflow = _make_mock_mlflow(registered_version="1", production_versions=[])
        mock_client = mock_mlflow.tracking.MlflowClient.return_value

        # Act
        with patch("app.training.trainer.mlflow", mock_mlflow):
            with patch.object(
                trainer, "save_model", return_value="models/model.pkl"
            ) as mock_save:
                trainer.run(data=gold_df)

        # Assert: promoted to Production
        transition_calls = mock_client.transition_model_version_stage.call_args_list
        stages_used = [c.kwargs.get("stage") or c.args[2] for c in transition_calls]
        assert "Production" in stages_used
        mock_save.assert_called_once()

    def test_gate_promotes_when_lift_meets_threshold(
        self, gold_df: pd.DataFrame, tmp_path: Any
    ) -> None:
        """Lift == threshold → promoted; champion archived (AC-9, AC-10)."""
        # Arrange: champion exists; gate sees lift == min_wape_lift_pct (2.0)
        champion = _make_champion(version="0", wape_tag="10.0")
        trainer = Trainer(config=REGISTRY_CONFIG)
        trainer.store_path = str(tmp_path / "models")
        mock_mlflow = _make_mock_mlflow(
            registered_version="1", production_versions=[champion]
        )
        mock_client = mock_mlflow.tracking.MlflowClient.return_value

        # Act: patch registry_gate.compute_wape_lift to return exactly threshold
        with patch("app.training.trainer.mlflow", mock_mlflow):
            with patch.object(
                trainer, "save_model", return_value="models/model.pkl"
            ) as mock_save:
                with patch(
                    "app.training.registry_gate.compute_wape_lift", return_value=2.0
                ):
                    trainer.run(data=gold_df)

        # Assert: challenger promoted, champion archived
        transition_calls = mock_client.transition_model_version_stage.call_args_list
        stages_used = [c.kwargs.get("stage") or c.args[2] for c in transition_calls]
        assert "Production" in stages_used
        assert "Archived" in stages_used
        mock_save.assert_called_once()

    def test_gate_writes_model_test_wape_tag_on_new_version(
        self, gold_df: pd.DataFrame, tmp_path: Any
    ) -> None:
        """set_model_version_tag called with key 'model_test_wape' and float string (user decision 1)."""
        # Arrange
        trainer = Trainer(config=REGISTRY_CONFIG)
        trainer.store_path = str(tmp_path / "models")
        mock_mlflow = _make_mock_mlflow(registered_version="1", production_versions=[])
        mock_client = mock_mlflow.tracking.MlflowClient.return_value

        # Act
        with patch("app.training.trainer.mlflow", mock_mlflow):
            trainer.run(data=gold_df)

        # Assert: model_test_wape tag was written (at minimum one call uses this key)
        tag_keys = [c.args[2] for c in mock_client.set_model_version_tag.call_args_list]
        assert "model_test_wape" in tag_keys
        # The value stored under model_test_wape must be float-parseable.
        for c in mock_client.set_model_version_tag.call_args_list:
            if c.args[2] == "model_test_wape":
                float(c.args[3])  # must be parseable as float

    def test_gate_result_tag_is_promoted_on_success(
        self, gold_df: pd.DataFrame, tmp_path: Any
    ) -> None:
        """mlflow.set_tag('gate_result', 'promoted') called on promotion (AC-16)."""
        # Arrange
        trainer = Trainer(config=REGISTRY_CONFIG)
        trainer.store_path = str(tmp_path / "models")
        mock_mlflow = _make_mock_mlflow(registered_version="1", production_versions=[])

        # Act
        with patch("app.training.trainer.mlflow", mock_mlflow):
            trainer.run(data=gold_df)

        # Assert
        mock_mlflow.set_tag.assert_any_call("gate_result", "promoted")

    # ------------------------------------------------------------------ #
    # Failure cases                                                        #
    # ------------------------------------------------------------------ #

    def test_gate_rejects_when_lift_below_threshold(
        self, gold_df: pd.DataFrame, tmp_path: Any
    ) -> None:
        """Lift < threshold → no stage transition; save_model NOT called (AC-11)."""
        # Arrange: champion exists; gate sees lift 1.0 (below threshold of 2.0)
        champion = _make_champion(version="0", wape_tag="10.0")
        trainer = Trainer(config=REGISTRY_CONFIG)
        trainer.store_path = str(tmp_path / "models")
        mock_mlflow = _make_mock_mlflow(
            registered_version="1", production_versions=[champion]
        )
        mock_client = mock_mlflow.tracking.MlflowClient.return_value

        # Act: patch registry_gate.compute_wape_lift to return below-threshold value
        with patch("app.training.trainer.mlflow", mock_mlflow):
            with patch(
                "app.training.registry_gate.compute_wape_lift", return_value=1.0
            ):
                with patch.object(trainer, "save_model") as mock_save:
                    trainer.run(data=gold_df)

        # Assert: no stage transition; disk unchanged
        mock_client.transition_model_version_stage.assert_not_called()
        mock_save.assert_not_called()

    def test_gate_sets_rejection_tags_when_rejected(
        self, gold_df: pd.DataFrame, tmp_path: Any
    ) -> None:
        """gate_result=rejected and gate_rejection_reason both set on rejection (AC-12)."""
        # Arrange
        champion = _make_champion(version="0", wape_tag="10.0")
        trainer = Trainer(config=REGISTRY_CONFIG)
        trainer.store_path = str(tmp_path / "models")
        mock_mlflow = _make_mock_mlflow(
            registered_version="1", production_versions=[champion]
        )

        # Act: gate sees lift 1.0 (below threshold)
        with patch("app.training.trainer.mlflow", mock_mlflow):
            with patch(
                "app.training.registry_gate.compute_wape_lift", return_value=1.0
            ):
                trainer.run(data=gold_df)

        # Assert
        tag_calls = {c.args[0]: c.args[1] for c in mock_mlflow.set_tag.call_args_list}
        assert tag_calls.get("gate_result") == "rejected"
        assert "gate_rejection_reason" in tag_calls
        assert len(tag_calls["gate_rejection_reason"]) > 0

    def test_gate_rejects_when_champion_wape_tag_missing(
        self, gold_df: pd.DataFrame, tmp_path: Any
    ) -> None:
        """Champion has no model_test_wape tag → rejection reason = champion_wape_unavailable."""
        # Arrange
        champion = _make_champion(version="0", wape_tag=None)
        trainer = Trainer(config=REGISTRY_CONFIG)
        trainer.store_path = str(tmp_path / "models")
        mock_mlflow = _make_mock_mlflow(
            registered_version="1", production_versions=[champion]
        )

        # Act
        with patch("app.training.trainer.mlflow", mock_mlflow):
            trainer.run(data=gold_df)

        # Assert
        tag_calls = {c.args[0]: c.args[1] for c in mock_mlflow.set_tag.call_args_list}
        assert tag_calls.get("gate_result") == "rejected"
        assert tag_calls.get("gate_rejection_reason") == "champion_wape_unavailable"

    def test_gate_rejects_with_no_baseline_when_incumbent_exists(
        self, tmp_path: Any
    ) -> None:
        """wape_lift_pct=None + Production exists → rejection with no_baseline_for_comparison (AC-14)."""
        # Arrange: use a tiny dataset so aligned_df is empty (no week-ago overlap)
        rng = np.random.RandomState(0)
        n = 100  # Only ~10 rows in test; no week-ago overlap possible
        hours = pd.date_range("2022-01-01", periods=n, freq="h")
        data: dict[str, Any] = {"hour": hours, "pulocationid": rng.randint(1, 5, n)}
        for feat in FEATURES:
            data[feat] = rng.rand(n)
        data["trip_count"] = rng.randint(1, 20, n)
        tiny_df = pd.DataFrame(data)

        champion = _make_champion(version="0", wape_tag="10.0")
        trainer = Trainer(config=REGISTRY_CONFIG)
        trainer.store_path = str(tmp_path / "models")
        mock_mlflow = _make_mock_mlflow(
            registered_version="1", production_versions=[champion]
        )

        # Act
        with patch("app.training.trainer.mlflow", mock_mlflow):
            with patch.object(trainer, "save_model") as mock_save:
                trainer.run(data=tiny_df)

        # Assert
        tag_calls = {c.args[0]: c.args[1] for c in mock_mlflow.set_tag.call_args_list}
        assert tag_calls.get("gate_result") == "rejected"
        assert tag_calls.get("gate_rejection_reason") == "no_baseline_for_comparison"
        mock_save.assert_not_called()

    def test_gate_promotes_when_no_baseline_and_no_incumbent(
        self, tmp_path: Any
    ) -> None:
        """wape_lift_pct=None + no Production → auto-promote (AC-15)."""
        # Arrange: tiny dataset → no week-ago overlap → wape_lift_pct=None
        rng = np.random.RandomState(0)
        n = 100
        hours = pd.date_range("2022-01-01", periods=n, freq="h")
        data: dict[str, Any] = {"hour": hours, "pulocationid": rng.randint(1, 5, n)}
        for feat in FEATURES:
            data[feat] = rng.rand(n)
        data["trip_count"] = rng.randint(1, 20, n)
        tiny_df = pd.DataFrame(data)

        trainer = Trainer(config=REGISTRY_CONFIG)
        trainer.store_path = str(tmp_path / "models")
        mock_mlflow = _make_mock_mlflow(registered_version="1", production_versions=[])

        # Act
        with patch("app.training.trainer.mlflow", mock_mlflow):
            with patch.object(
                trainer, "save_model", return_value="models/model.pkl"
            ) as mock_save:
                trainer.run(data=tiny_df)

        # Assert: promoted despite no baseline
        tag_calls = {c.args[0]: c.args[1] for c in mock_mlflow.set_tag.call_args_list}
        assert tag_calls.get("gate_result") == "promoted"
        mock_save.assert_called_once()

    # ------------------------------------------------------------------ #
    # Edge cases                                                           #
    # ------------------------------------------------------------------ #

    def test_gate_passes_when_lift_exactly_equals_threshold(
        self, gold_df: pd.DataFrame, tmp_path: Any
    ) -> None:
        """Lift exactly equal to min_wape_lift_pct is treated as pass (>= not >) (AC-9 boundary)."""
        # Arrange: champion exists; gate sees lift == 2.0 (exactly threshold)
        champion = _make_champion(version="0", wape_tag="10.0")
        trainer = Trainer(config=REGISTRY_CONFIG)
        trainer.store_path = str(tmp_path / "models")
        mock_mlflow = _make_mock_mlflow(
            registered_version="1", production_versions=[champion]
        )

        # Act: patch registry_gate.compute_wape_lift to return exactly threshold
        with patch("app.training.trainer.mlflow", mock_mlflow):
            with patch(
                "app.training.registry_gate.compute_wape_lift", return_value=2.0
            ):
                with patch.object(
                    trainer, "save_model", return_value="models/model.pkl"
                ) as mock_save:
                    trainer.run(data=gold_df)

        # Assert: promoted (not rejected)
        tag_calls = {c.args[0]: c.args[1] for c in mock_mlflow.set_tag.call_args_list}
        assert tag_calls.get("gate_result") == "promoted"
        mock_save.assert_called_once()

    def test_trainer_reads_tracking_uri_from_config(self) -> None:
        """mlflow.set_tracking_uri called with value from config (AC-4)."""
        # Arrange
        with patch("app.training.trainer.mlflow") as mock_mlflow:
            # Act: set_tracking_uri is called in __init__
            Trainer(config=REGISTRY_CONFIG)

        # Assert: called with the configured URI, not a hardcoded path
        mock_mlflow.set_tracking_uri.assert_called_once_with("sqlite:///mlflow.db")

    def test_gate_falls_back_to_default_model_name_when_key_missing(
        self, gold_df: pd.DataFrame, tmp_path: Any
    ) -> None:
        """mlflow.model_name absent → register_model called with 'ny_taxi_demand_model' (AC-2)."""
        # Arrange: config without mlflow.model_name
        cfg_no_name: dict[str, Any] = {
            **REGISTRY_CONFIG,
            "mlflow": {
                "experiment_name": "test_experiment",
                "tracking_uri": "sqlite:///mlflow.db",
                # model_name intentionally absent
            },
        }
        trainer = Trainer(config=cfg_no_name)
        trainer.store_path = str(tmp_path / "models")
        mock_mlflow = _make_mock_mlflow(registered_version="1", production_versions=[])

        # Act
        with patch("app.training.trainer.mlflow", mock_mlflow):
            trainer.run(data=gold_df)

        # Assert: default model name used
        pos_args = mock_mlflow.register_model.call_args.args
        assert pos_args[1] == "ny_taxi_demand_model"

    def test_gate_falls_back_to_default_min_lift_when_key_missing(
        self, gold_df: pd.DataFrame, tmp_path: Any
    ) -> None:
        """train.gate absent → min_wape_lift_pct defaults to 2.0 (AC-3)."""
        # Arrange: config without train.gate
        cfg_no_gate: dict[str, Any] = {
            **REGISTRY_CONFIG,
            "train": {
                "features": FEATURES,
                "target": "trip_count",
                "split_ratios": [0.8, 0.1, 0.1],
                # gate intentionally absent
            },
        }
        trainer = Trainer(config=cfg_no_gate)

        # Assert: default 2.0 was read
        assert trainer.min_wape_lift_pct == 2.0

        trainer.store_path = str(tmp_path / "models")
        champion = _make_champion(version="0", wape_tag="10.0")
        mock_mlflow = _make_mock_mlflow(
            registered_version="1", production_versions=[champion]
        )

        # Lift below default threshold of 2.0 → rejected
        with patch("app.training.trainer.mlflow", mock_mlflow):
            with patch(
                "app.training.registry_gate.compute_wape_lift", return_value=1.0
            ):
                result = trainer.run(data=gold_df)

        tag_calls = {c.args[0]: c.args[1] for c in mock_mlflow.set_tag.call_args_list}
        assert tag_calls.get("gate_result") == "rejected"
        assert result["model_path"] is None

    def test_gate_writes_wape_lift_pct_tag_on_promotion(
        self, gold_df: pd.DataFrame, tmp_path: Any
    ) -> None:
        """set_model_version_tag called with 'wape_lift_pct' and float string on promotion (AC #11)."""
        # Arrange: champion exists so apply_gate is called; patch lift to pass gate.
        champion = _make_champion(version="0", wape_tag="10.0")
        trainer = Trainer(config=REGISTRY_CONFIG)
        trainer.store_path = str(tmp_path / "models")
        mock_mlflow = _make_mock_mlflow(
            registered_version="1", production_versions=[champion]
        )
        mock_client = mock_mlflow.tracking.MlflowClient.return_value

        # Act: lift 5.0 >= threshold 2.0 → promoted
        with patch("app.training.trainer.mlflow", mock_mlflow):
            with patch(
                "app.training.registry_gate.compute_wape_lift", return_value=5.0
            ):
                with patch.object(
                    trainer, "save_model", return_value="models/model.pkl"
                ):
                    trainer.run(data=gold_df)

        # Assert: wape_lift_pct tag written with a float-parseable string value.
        tag_keys = [c.args[2] for c in mock_client.set_model_version_tag.call_args_list]
        assert "wape_lift_pct" in tag_keys
        for c in mock_client.set_model_version_tag.call_args_list:
            if c.args[2] == "wape_lift_pct":
                float(c.args[3])  # value must be float-parseable

    def test_gate_does_not_write_wape_lift_pct_tag_when_none(
        self, tmp_path: Any
    ) -> None:
        """wape_lift_pct=None (no baseline overlap) → wape_lift_pct tag not written (AC #12)."""
        # Arrange: tiny dataset → no week-ago overlap → wape_lift_pct=None
        # AND no Production incumbent → auto-promote path (gate_result="promoted")
        rng = np.random.RandomState(0)
        n = 100
        hours = pd.date_range("2022-01-01", periods=n, freq="h")
        data: dict[str, Any] = {"hour": hours, "pulocationid": rng.randint(1, 5, n)}
        for feat in FEATURES:
            data[feat] = rng.rand(n)
        data["trip_count"] = rng.randint(1, 20, n)
        tiny_df = pd.DataFrame(data)

        trainer = Trainer(config=REGISTRY_CONFIG)
        trainer.store_path = str(tmp_path / "models")
        mock_mlflow = _make_mock_mlflow(registered_version="1", production_versions=[])
        mock_client = mock_mlflow.tracking.MlflowClient.return_value

        # Act: auto-promoted but wape_lift_pct is None
        with patch("app.training.trainer.mlflow", mock_mlflow):
            with patch.object(trainer, "save_model", return_value="models/model.pkl"):
                result = trainer.run(data=tiny_df)

        # Assert: wape_lift_pct was None and its tag was NOT written
        assert result["wape_lift_pct"] is None
        tag_keys = [c.args[2] for c in mock_client.set_model_version_tag.call_args_list]
        assert "wape_lift_pct" not in tag_keys

    def test_gate_does_not_write_wape_lift_pct_tag_when_inf(
        self, gold_df: pd.DataFrame, tmp_path: Any
    ) -> None:
        """wape_lift_pct=inf → wape_lift_pct tag not written (edge case).

        Both trainer and registry_gate copies of compute_wape_lift are patched
        because trainer.py calls its own import to set wape_lift_pct while
        registry_gate.py calls its own import inside apply_gate.
        Patching app.training.trainer.compute_wape_lift ensures the local
        wape_lift_pct variable is inf; patching registry_gate ensures the gate
        still promotes (inf >= 2.0).
        """
        # Arrange: champion exists; both compute_wape_lift copies return inf.
        champion = _make_champion(version="0", wape_tag="10.0")
        trainer = Trainer(config=REGISTRY_CONFIG)
        trainer.store_path = str(tmp_path / "models")
        mock_mlflow = _make_mock_mlflow(
            registered_version="1", production_versions=[champion]
        )
        mock_client = mock_mlflow.tracking.MlflowClient.return_value

        with patch("app.training.trainer.mlflow", mock_mlflow):
            with patch(
                "app.training.trainer.compute_wape_lift",
                return_value=float("inf"),
            ):
                with patch(
                    "app.training.registry_gate.compute_wape_lift",
                    return_value=float("inf"),
                ):
                    with patch.object(
                        trainer, "save_model", return_value="models/model.pkl"
                    ):
                        result = trainer.run(data=gold_df)

        # Assert: wape_lift_pct was inf and the tag was NOT written
        assert result["wape_lift_pct"] == float("inf")
        tag_keys = [c.args[2] for c in mock_client.set_model_version_tag.call_args_list]
        assert "wape_lift_pct" not in tag_keys

    def test_register_model_does_not_raise_on_sqlite_uri(
        self, gold_df: pd.DataFrame, tmp_path: Any
    ) -> None:
        """AC-5: mlflow.register_model succeeds (no MlflowException) when the
        tracking URI is sqlite:///mlflow.db.

        The boundary mock configures register_model to return a normal version
        object (the default MagicMock behaviour), which is what the real SQLite
        backend does.  The test confirms that Trainer.run() completes without
        raising and that register_model was called exactly once — documenting
        the contractual dependency on the sqlite URI supporting registry calls.
        """
        # Arrange
        trainer = Trainer(config=REGISTRY_CONFIG)
        trainer.store_path = str(tmp_path / "models")
        mock_mlflow = _make_mock_mlflow(registered_version="1", production_versions=[])

        # Act — must not raise MlflowException or any other exception
        with patch("app.training.trainer.mlflow", mock_mlflow):
            result = trainer.run(data=gold_df)

        # Assert: register_model was called once and run completed cleanly
        mock_mlflow.register_model.assert_called_once()
        assert result is not None


# -- Import-consolidation regression tests ----------------------------------


class TestMetricsConsolidation:
    """Confirm that Trainer, Predictor, and HourlySimulator all resolve
    to app.core.metrics.evaluate under the hood."""

    def test_trainer_evaluate_delegates_to_core_metrics(self) -> None:
        """Trainer.evaluate wraps app.core.metrics.evaluate."""
        from app.training.trainer import Trainer as _Trainer

        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        trainer_result = _Trainer.evaluate(y_true, y_pred)
        core_result = core_evaluate(y_true, y_pred)

        assert trainer_result == core_result

    def test_predictor_evaluate_delegates_to_core_metrics(self) -> None:
        """Predictor.evaluate wraps app.core.metrics.evaluate."""
        from app.inference.predictor import Predictor

        y_true = pd.Series([4.0, 5.0, 6.0])
        y_pred = np.array([4.0, 5.0, 6.0])

        predictor_result = Predictor.evaluate(y_true, y_pred)
        core_result = core_evaluate(y_true, y_pred)

        assert predictor_result == core_result

    def test_simulator_compute_metrics_delegates_to_core_metrics(self) -> None:
        """HourlySimulator.compute_metrics wraps app.core.metrics.evaluate."""
        from app.inference.simulator import HourlySimulator

        actuals = np.array([7.0, 8.0, 9.0])
        preds = np.array([7.0, 8.0, 9.0])

        sim_result = HourlySimulator.compute_metrics(actuals, preds)
        core_result = core_evaluate(pd.Series(actuals), preds)

        assert sim_result == core_result
