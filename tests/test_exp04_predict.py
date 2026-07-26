"""Tests for orchestration_experiment/dags/exp04_predict.py.

PBI EXP-04 (predict + log) — the pure predict helper the DAG delegates to.
Inserts a ``predict`` task between ``feature`` and ``log_predict``.

Two seams keep this unit-testable:

  * ``build_predict_config`` — deep-overrides ``data.gold_path``,
    ``model.store_path``, and ``inference.output_path`` to point at the
    experiment roots. The base config is never mutated.
  * ``run_predict`` — constructs ``Predictor(config=config)``, calls ``.run()``,
    and returns the ``output_path`` string from the response.

AC coverage:
  - build_predict_config overrides the three experiment paths
  - build_predict_config does not mutate the base config (deep copy)
  - build_predict_config preserves unrelated keys
  - run_predict constructs Predictor with the injected config and calls run()
  - run_predict returns the output_path from the Predictor response
"""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import patch

import pytest

from orchestration_experiment.dags.exp04_predict import (
    build_predict_config,
    run_predict,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

GOLD_ROOT = "/opt/airflow/data_exp/gold/yellow_tripdata"
MODEL_ROOT = "/opt/airflow/data_exp/models_trained"
PREDICTIONS_PATH = "/opt/airflow/data_exp/predictions/demand_predictions.parquet"


@pytest.fixture
def base_config() -> dict[str, Any]:
    """Minimal config mirroring the shape Predictor reads."""
    return {
        "env": "experiment",
        "data": {
            "bronze_path": "data/bronze/yellow_tripdata/bronze.parquet",
            "silver_path": "data/silver/yellow_tripdata_cleaned.parquet",
            "gold_path": "data/gold/yellow_tripdata",
        },
        "model": {"store_path": "models/"},
        "train": {
            "features": ["trip_count_lag1"],
            "target": "target_sum_t1_t2",
        },
        "inference": {
            "output_path": "data/predictions/demand_predictions.parquet",
            "simulation": {"start_hour": "2021-03-01 00:00:00"},
        },
    }


# ---------------------------------------------------------------------------
# TestBuildPredictConfig
# ---------------------------------------------------------------------------


class TestBuildPredictConfig:
    """build_predict_config injects the isolated paths (the seam)."""

    def test_overrides_paths_without_mutating_base(
        self, base_config: dict[str, Any]
    ) -> None:
        """AC: overrides three paths AND base config is unchanged (deep-copy contract)."""
        # Arrange
        snapshot = copy.deepcopy(base_config)

        # Act
        result = build_predict_config(
            base_config,
            gold_root=GOLD_ROOT,
            model_root=MODEL_ROOT,
            predictions_path=PREDICTIONS_PATH,
        )

        # Assert — overrides applied
        assert result["data"]["gold_path"] == GOLD_ROOT
        assert result["model"]["store_path"] == MODEL_ROOT
        assert result["inference"]["output_path"] == PREDICTIONS_PATH
        # Assert — base not mutated
        assert base_config == snapshot

    def test_preserves_unrelated_keys(self, base_config: dict[str, Any]) -> None:
        """Only the redirected keys change; everything else passes through."""
        # Act
        result = build_predict_config(
            base_config,
            gold_root=GOLD_ROOT,
            model_root=MODEL_ROOT,
            predictions_path=PREDICTIONS_PATH,
        )

        # Assert
        assert result["data"]["bronze_path"] == base_config["data"]["bronze_path"]
        assert result["train"]["features"] == base_config["train"]["features"]
        assert result["train"]["target"] == base_config["train"]["target"]
        assert (
            result["inference"]["simulation"]["start_hour"]
            == base_config["inference"]["simulation"]["start_hour"]
        )


# ---------------------------------------------------------------------------
# TestRunPredict
# ---------------------------------------------------------------------------


class TestRunPredict:
    """run_predict wires Predictor with the override config."""

    def test_calls_predictor_with_config_and_returns_output_path(
        self, base_config: dict[str, Any]
    ) -> None:
        """AC: Predictor constructed with config=cfg, .run() called, returns output_path."""
        # Arrange
        cfg = build_predict_config(
            base_config,
            gold_root=GOLD_ROOT,
            model_root=MODEL_ROOT,
            predictions_path=PREDICTIONS_PATH,
        )
        fake_response = {
            "metrics": {"mae": 1.0},
            "output_path": PREDICTIONS_PATH,
            "n_predictions": 10,
        }

        with patch(
            "orchestration_experiment.dags.exp04_predict.Predictor"
        ) as mock_predictor_cls:
            mock_predictor_cls.return_value.run.return_value = fake_response

            # Act
            result = run_predict(config=cfg)

        # Assert — Predictor constructed with the override config
        mock_predictor_cls.assert_called_once_with(config=cfg)
        mock_predictor_cls.return_value.run.assert_called_once_with()
        # Assert — returns the output_path string
        assert result == PREDICTIONS_PATH
