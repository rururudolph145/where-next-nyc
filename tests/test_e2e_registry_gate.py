"""Acceptance tests for PBI 20 — MLflow Model Registry + Champion-vs-Challenger Gate.

Story: As a pipeline operator, I want the training pipeline to register each trained
model in the MLflow Model Registry and automatically promote it to Production only when
it beats the current champion's test WAPE by a configurable margin, so that the
Production model slot always holds a validated improvement and regressions are blocked
without manual intervention.

These tests exercise Trainer.run() end-to-end on a synthetic Gold dataset with MLflow
registry operations mocked at the boundary.  Observable outcomes are asserted —
what tags exist on the run, whether the disk pickle was written — rather than which
mock methods were called with which arguments.

Acceptance criteria covered here (behavioural scenarios):
  Scenario A → AC-8, AC-10, AC-16  (first-ever model auto-promoted)
  Scenario B → AC-6, AC-9, AC-10, AC-16  (challenger beats champion → promoted)
  Scenario C → AC-11, AC-12, AC-13, AC-16  (worse challenger → rejected, disk intact)
  Scenario D → AC-14, AC-16  (no baseline overlap + incumbent → rejected, disk intact)
  Scenario E → AC-15  (no baseline overlap, no incumbent → auto-promoted)
  Edge: boundary → AC-9 (lift exactly equals threshold → still promoted)

  AC-1, AC-2, AC-3 → config file values confirmed by direct yaml read (below).
  AC-4 → covered by unit test test_trainer_reads_tracking_uri_from_config in
      test_train.py::TestRegistryGate — set_tracking_uri in __init__ is an
      implementation-level assertion better suited to a unit test.
  AC-5 → covered by unit test test_register_model_does_not_raise_on_sqlite_uri in
      test_train.py::TestRegistryGate — documents that register_model succeeds
      (no MlflowException) when the tracking URI is sqlite:///mlflow.db.
  AC-7 (source URI) → covered by unit tests.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from mlflow.exceptions import MlflowException

from app.core.config import AppConfig
from app.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Synthetic Gold fixture — reuses the 4-week pattern from PBI 18 e2e tests
# ---------------------------------------------------------------------------

_BASE_TS = pd.Timestamp("2022-01-03 00:00:00")  # Monday — weekly pattern anchors here
_ZONES = [1, 2, 3]

# Feature columns used by the acceptance-test config (matches test_train.py::FEATURES)
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


def _make_4week_gold(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    """Build a 4-week × 24-hour × 3-zone Gold parquet with week-ago overlap.

    trip_count = zone * 10 + hour_of_day (perfectly repeating weekly pattern)
    so LastWeekSameHourBaseline has non-empty aligned_df in the test split.

    Writes the parquet to ``tmp_path / "gold" / "hourly_demand_featured.parquet"``.

    Returns:
        (gold_dir_path, full_gold_df) tuple.
    """
    records = []
    for week in range(4):
        for hour in range(24):
            ts = _BASE_TS + pd.Timedelta(weeks=week, hours=hour)
            for zone in _ZONES:
                records.append(
                    {
                        "pulocationid": zone,
                        "hour": ts,
                        "trip_count": float(zone * 10 + hour),
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


def _make_no_overlap_gold(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    """Build a 1-week Gold fixture so the test split has no week-ago overlap.

    With a 80/10/10 split the test slice (last 10%) spans < 7 days from the
    start of the Gold data, so LastWeekSameHourBaseline returns an empty
    aligned_df — ensuring wape_lift_pct is None in Trainer.run().

    Returns:
        (gold_dir_path, short_gold_df) tuple.
    """
    rng = np.random.RandomState(99)
    n = 100  # ~10 test rows, no week-ago overlap possible
    hours = pd.date_range("2022-01-03 00:00:00", periods=n, freq="h")
    data: dict[str, Any] = {
        "pulocationid": rng.randint(1, 4, n),
        "hour": hours,
        "trip_count": rng.randint(1, 20, n).astype(float),
    }
    for feat in _FEATURES:
        data[feat] = rng.rand(n)

    short_df = pd.DataFrame(data)
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir(parents=True)
    short_df.to_parquet(gold_dir / "hourly_demand_featured.parquet", index=False)
    return gold_dir, short_df


# ---------------------------------------------------------------------------
# Config factory — points all paths at tmp_path, includes PBI-20 registry keys
# ---------------------------------------------------------------------------


def _registry_config(tmp_path: Path) -> AppConfig:
    """Return a full Trainer AppConfig with PBI-20 keys, all paths in tmp_path."""
    from tests.conftest import _FULL_CONFIG_DICT

    base = AppConfig.model_validate(_FULL_CONFIG_DICT)
    return base.model_copy(
        update={
            "data": base.data.model_copy(update={"gold_path": str(tmp_path / "gold")}),
            "train": base.train.model_copy(
                update={
                    "features": _FEATURES,
                    "target": "trip_count",
                    "split_ratios": [0.8, 0.1, 0.1],
                }
            ),
            "model": base.model.model_copy(
                update={
                    "params": {
                        "n_estimators": 5,
                        "learning_rate": 0.1,
                        "num_leaves": 4,
                        "random_state": 42,
                        "verbosity": -1,
                    },
                    "store_path": str(tmp_path / "models"),
                }
            ),
            "mlflow": base.mlflow.model_copy(
                update={
                    "experiment_name": "test_pbi20_acceptance",
                    "tracking_uri": "sqlite:///mlflow.db",
                }
            ),
        }
    )


# ---------------------------------------------------------------------------
# Mock MLflow boundary helpers
# ---------------------------------------------------------------------------


def _build_mlflow_mock(
    run_id: str = "abc123",
    registered_version: str = "1",
    production_versions: list[Any] | None = None,
) -> MagicMock:
    """Construct a boundary mock for app.training.trainer.mlflow.

    Configures:
    - start_run context manager returning an active_run with a stable run_id
    - register_model returning a ModelVersion-like object with .version
    - tracking.MlflowClient() returning a mock client with configurable
      get_latest_versions return value
    """
    mock_mlflow = MagicMock()

    # Context manager: ``with mlflow.start_run() as active_run:``
    active_run = MagicMock()
    active_run.info.run_id = run_id
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=active_run)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    # register_model → ModelVersion-like object
    mock_version = MagicMock()
    mock_version.version = registered_version
    mock_mlflow.register_model.return_value = mock_version

    # MlflowClient instance returned via mlflow.tracking.MlflowClient()
    mock_client = MagicMock()
    if production_versions:
        mock_client.get_model_version_by_alias.return_value = production_versions[0]
    else:
        mock_client.get_model_version_by_alias.side_effect = MlflowException(
            "Alias champion not found"
        )
    mock_mlflow.tracking.MlflowClient.return_value = mock_client

    return mock_mlflow


def _champion_mock(version: str = "0", wape_tag: str | None = "15.0") -> MagicMock:
    """Return a ModelVersion mock representing the current Production champion."""
    champion = MagicMock()
    champion.version = version
    champion.tags = {"model_test_wape": wape_tag} if wape_tag is not None else {}
    return champion


def _extract_run_tags(mock_mlflow: MagicMock) -> dict[str, str]:
    """Extract all (key, value) pairs passed to mlflow.set_tag as a dict.

    This reads the observable state of the MLflow run (which tags were written)
    without caring which call order they were written in.
    """
    return {call.args[0]: call.args[1] for call in mock_mlflow.set_tag.call_args_list}


# ---------------------------------------------------------------------------
# PBI-20 Acceptance Tests
# ---------------------------------------------------------------------------


class TestRegistryGateAcceptance:
    """Story-level acceptance tests for PBI-20: registry gate scenarios.

    Each test corresponds to one primary behavioural scenario from the user
    story and covers one or more acceptance criteria.  Tests assert on
    observable outcomes (disk files, run tags, return dict values) not on
    mock call shapes.
    """

    # ------------------------------------------------------------------
    # AC-1 / AC-2 / AC-3 — Config file contains required keys
    # ------------------------------------------------------------------

    def test_ac1_config_yml_has_sqlite_tracking_uri(self) -> None:
        """AC-1: config.yml tracking_uri uses a SQLite backend under data/."""
        # Arrange
        config_path = Path(__file__).parent.parent / "config.yml"

        # Act
        with open(config_path) as fh:
            config = yaml.safe_load(fh)

        # Assert — canonical path is data/mlflow.db so the container volume mount
        # (../data → /opt/airflow/data) can reach it from both host and container.
        assert config["mlflow"]["tracking_uri"] == "sqlite:///data/mlflow.db"

    def test_ac2_config_yml_has_model_name_key(self) -> None:
        """AC-2: config.yml contains mlflow.model_name = ny_taxi_demand_model."""
        # Arrange
        config_path = Path(__file__).parent.parent / "config.yml"

        # Act
        with open(config_path) as fh:
            config = yaml.safe_load(fh)

        # Assert
        assert config["mlflow"]["model_name"] == "ny_taxi_demand_model"

    def test_ac3_config_yml_has_gate_min_wape_lift_pct(self) -> None:
        """AC-3: config.yml contains train.gate.min_wape_lift_pct = 2.0."""
        # Arrange
        config_path = Path(__file__).parent.parent / "config.yml"

        # Act
        with open(config_path) as fh:
            config = yaml.safe_load(fh)

        # Assert
        assert config["train"]["gate"]["min_wape_lift_pct"] == pytest.approx(2.0)

    # ------------------------------------------------------------------
    # Scenario A — First-ever model: auto-promoted (AC-8, AC-10, AC-16)
    # ------------------------------------------------------------------

    def test_scenario_a_first_ever_model_is_auto_promoted(self, tmp_path: Path) -> None:
        """AC-8, AC-10, AC-16: When no Production version exists the challenger
        is auto-promoted, the disk pickle is written, and the run tag
        gate_result is 'promoted'.

        Observable outcomes:
        - result["model_path"] points to an existing file on disk.
        - The MLflow run has tag gate_result == 'promoted'.
        - No gate_rejection_reason tag is set.
        """
        # Arrange
        _, gold_df = _make_4week_gold(tmp_path)
        config = _registry_config(tmp_path)
        trainer = Trainer(config=config)
        mock_mlflow = _build_mlflow_mock(
            run_id="first_run", registered_version="1", production_versions=[]
        )

        # Act
        with patch("app.training.trainer.mlflow", mock_mlflow):
            result = trainer.run(data=gold_df)

        # Assert — AC-10: disk pickle written
        assert result["model_path"] is not None, (
            "model_path should not be None when gate promotes"
        )
        assert os.path.exists(result["model_path"]), (
            f"model.pkl not found at {result['model_path']}"
        )

        # Assert — AC-16: run tag gate_result is present
        run_tags = _extract_run_tags(mock_mlflow)
        assert run_tags.get("gate_result") == "promoted", (
            f"Expected gate_result='promoted', got: {run_tags.get('gate_result')}"
        )

        # Assert — AC-8: no rejection reason (auto-promote path)
        assert "gate_rejection_reason" not in run_tags, (
            "Auto-promote should not set gate_rejection_reason"
        )

    # ------------------------------------------------------------------
    # Scenario B — Challenger beats champion: promoted (AC-6, AC-9, AC-10, AC-16)
    # ------------------------------------------------------------------

    def test_scenario_b_better_challenger_is_promoted(self, tmp_path: Path) -> None:
        """AC-9, AC-10, AC-16: When the challenger's WAPE lift >= min_wape_lift_pct
        the challenger is promoted to Production, the disk pickle is written, and
        the run has gate_result == 'promoted'.

        The champion WAPE is set to a very high value (90.0%) so any real
        model will comfortably beat the threshold.  The gate runs real
        compute_wape_lift logic (not mocked) against the champion tag.

        Observable outcomes:
        - result["model_path"] is not None and the file exists on disk.
        - Run tag gate_result == 'promoted'.
        - No gate_rejection_reason on the run.
        """
        # Arrange: champion with very high WAPE (90%) — easy for any model to beat
        champion = _champion_mock(version="0", wape_tag="90.0")
        _, gold_df = _make_4week_gold(tmp_path)
        config = _registry_config(tmp_path)
        trainer = Trainer(config=config)
        mock_mlflow = _build_mlflow_mock(
            run_id="challenger_run",
            registered_version="2",
            production_versions=[champion],
        )

        # Act
        with patch("app.training.trainer.mlflow", mock_mlflow):
            result = trainer.run(data=gold_df)

        # Assert — AC-10: disk pickle written on promotion
        assert result["model_path"] is not None, (
            "model_path should not be None when challenger is promoted"
        )
        assert os.path.exists(result["model_path"]), (
            f"model.pkl not found at {result['model_path']}"
        )

        # Assert — AC-16 + AC-9: gate_result is 'promoted'
        run_tags = _extract_run_tags(mock_mlflow)
        assert run_tags.get("gate_result") == "promoted", (
            f"Expected gate_result='promoted', got: {run_tags.get('gate_result')}"
        )
        assert "gate_rejection_reason" not in run_tags, (
            "Promoted run should not have gate_rejection_reason"
        )

    # ------------------------------------------------------------------
    # Scenario C — Worse challenger: rejected, disk intact (AC-11, AC-12, AC-13, AC-16)
    # ------------------------------------------------------------------

    def test_scenario_c_worse_challenger_is_rejected_disk_unchanged(
        self, tmp_path: Path
    ) -> None:
        """AC-11, AC-12, AC-13, AC-16: When the challenger's WAPE lift falls below
        min_wape_lift_pct the challenger is NOT promoted, the disk pickle is NOT
        written (or overwritten), and the run has gate_result == 'rejected' with
        a non-empty gate_rejection_reason.

        The champion WAPE is set to a very low value (0.5%) so the lift from any
        realistic model will be negative (well below the 2.0% threshold), and we
        also patch compute_wape_lift in registry_gate to return 1.0 (< 2.0) to
        make the test deterministic regardless of dataset behaviour.

        Observable outcomes:
        - result["model_path"] is None (AC-13: disk not overwritten).
        - No model.pkl written in tmp_path/models (AC-13 verified on disk).
        - Run tag gate_result == 'rejected' (AC-12, AC-16).
        - Run tag gate_rejection_reason is a non-empty string (AC-12).
        """
        # Arrange: champion with very low WAPE — challenger cannot beat it by 2%
        champion = _champion_mock(version="1", wape_tag="0.5")
        _, gold_df = _make_4week_gold(tmp_path)
        config = _registry_config(tmp_path)
        trainer = Trainer(config=config)
        mock_mlflow = _build_mlflow_mock(
            run_id="weak_challenger",
            registered_version="2",
            production_versions=[champion],
        )

        # Act — patch compute_wape_lift in registry_gate to return sub-threshold lift
        with patch("app.training.trainer.mlflow", mock_mlflow):
            with patch(
                "app.training.registry_gate.compute_wape_lift", return_value=1.0
            ):
                result = trainer.run(data=gold_df)

        # Assert — AC-13: disk model NOT written
        assert result["model_path"] is None, (
            "model_path must be None when gate rejects the challenger"
        )
        model_pkl = tmp_path / "models" / "model.pkl"
        assert not model_pkl.exists(), (
            "model.pkl must NOT be written to disk when challenger is rejected"
        )

        # Assert — AC-12 + AC-16: run tags
        run_tags = _extract_run_tags(mock_mlflow)
        assert run_tags.get("gate_result") == "rejected", (
            f"Expected gate_result='rejected', got: {run_tags.get('gate_result')}"
        )
        rejection_reason = run_tags.get("gate_rejection_reason", "")
        assert len(rejection_reason) > 0, (
            "gate_rejection_reason must be a non-empty string on rejection"
        )

    # ------------------------------------------------------------------
    # Scenario D — No baseline overlap + incumbent: rejected (AC-14, AC-16)
    # ------------------------------------------------------------------

    def test_scenario_d_no_baseline_with_incumbent_is_rejected(
        self, tmp_path: Path
    ) -> None:
        """AC-14, AC-16: When wape_lift_pct is None (no week-ago overlap) and a
        Production champion already exists, the challenger is rejected with
        gate_rejection_reason == 'no_baseline_for_comparison', and the disk
        pickle is NOT written.

        Observable outcomes:
        - result["wape_lift_pct"] is None (confirms no baseline overlap).
        - result["model_path"] is None (AC-14: disk preserved).
        - No model.pkl in tmp_path/models.
        - Run tag gate_result == 'rejected' (AC-16).
        - Run tag gate_rejection_reason == 'no_baseline_for_comparison' (AC-14).
        """
        # Arrange: short gold → no week-ago overlap → wape_lift_pct=None
        _, short_df = _make_no_overlap_gold(tmp_path)
        champion = _champion_mock(version="1", wape_tag="20.0")
        config = _registry_config(tmp_path)
        trainer = Trainer(config=config)
        mock_mlflow = _build_mlflow_mock(
            run_id="no_baseline_run",
            registered_version="2",
            production_versions=[champion],
        )

        # Act
        with patch("app.training.trainer.mlflow", mock_mlflow):
            result = trainer.run(data=short_df)

        # Assert — confirm the no-overlap condition held (precondition check)
        assert result["wape_lift_pct"] is None, (
            "Fixture should produce no week-ago overlap; wape_lift_pct must be None"
        )

        # Assert — AC-14: disk pickle NOT written
        assert result["model_path"] is None, (
            "model_path must be None when no baseline + incumbent exists"
        )
        model_pkl = tmp_path / "models" / "model.pkl"
        assert not model_pkl.exists(), (
            "model.pkl must NOT be written to disk on no_baseline_for_comparison rejection"
        )

        # Assert — AC-14 + AC-16: run tags
        run_tags = _extract_run_tags(mock_mlflow)
        assert run_tags.get("gate_result") == "rejected", (
            f"Expected gate_result='rejected', got: {run_tags.get('gate_result')}"
        )
        assert run_tags.get("gate_rejection_reason") == "no_baseline_for_comparison", (
            f"Expected reason 'no_baseline_for_comparison', "
            f"got: {run_tags.get('gate_rejection_reason')}"
        )

    # ------------------------------------------------------------------
    # Scenario E — No baseline overlap, no incumbent: auto-promoted (AC-15)
    # ------------------------------------------------------------------

    def test_scenario_e_no_baseline_no_incumbent_is_auto_promoted(
        self, tmp_path: Path
    ) -> None:
        """AC-15: When wape_lift_pct is None (no week-ago overlap) AND no Production
        champion exists, the AC-8 rule applies: auto-promote regardless.

        Observable outcomes:
        - result["wape_lift_pct"] is None (confirms no baseline overlap).
        - result["model_path"] is not None and the file exists on disk (AC-10 via AC-8).
        - Run tag gate_result == 'promoted' (AC-16).
        """
        # Arrange: short gold → wape_lift_pct=None; empty registry
        _, short_df = _make_no_overlap_gold(tmp_path)
        config = _registry_config(tmp_path)
        trainer = Trainer(config=config)
        mock_mlflow = _build_mlflow_mock(
            run_id="first_run_no_baseline",
            registered_version="1",
            production_versions=[],
        )

        # Act
        with patch("app.training.trainer.mlflow", mock_mlflow):
            result = trainer.run(data=short_df)

        # Assert — confirm no-overlap precondition
        assert result["wape_lift_pct"] is None, (
            "Fixture should produce no week-ago overlap; wape_lift_pct must be None"
        )

        # Assert — AC-15 + AC-10: disk pickle written on auto-promote
        assert result["model_path"] is not None, (
            "model_path must not be None when first-ever run is auto-promoted "
            "despite missing baseline"
        )
        assert os.path.exists(result["model_path"]), (
            f"model.pkl not found at {result['model_path']}"
        )

        # Assert — AC-16: gate_result tag
        run_tags = _extract_run_tags(mock_mlflow)
        assert run_tags.get("gate_result") == "promoted", (
            f"Expected gate_result='promoted', got: {run_tags.get('gate_result')}"
        )

    # ------------------------------------------------------------------
    # Edge: lift exactly equals threshold — still promoted (AC-9 boundary)
    # ------------------------------------------------------------------

    def test_edge_lift_exactly_at_threshold_is_promoted(self, tmp_path: Path) -> None:
        """AC-9 boundary (>= not >): when compute_wape_lift returns exactly
        min_wape_lift_pct (2.0) the challenger is promoted, not rejected.

        compute_wape_lift in registry_gate is patched to return exactly 2.0
        so the boundary condition is deterministic regardless of dataset.

        Observable outcomes:
        - result["model_path"] is not None and the file exists on disk.
        - Run tag gate_result == 'promoted'.
        """
        # Arrange: champion exists so the gate comparison path is exercised
        champion = _champion_mock(version="1", wape_tag="10.0")
        _, gold_df = _make_4week_gold(tmp_path)
        config = _registry_config(tmp_path)
        trainer = Trainer(config=config)
        mock_mlflow = _build_mlflow_mock(
            run_id="boundary_run",
            registered_version="2",
            production_versions=[champion],
        )

        # Act — patch compute_wape_lift to return exactly the threshold
        with patch("app.training.trainer.mlflow", mock_mlflow):
            with patch(
                "app.training.registry_gate.compute_wape_lift", return_value=2.0
            ):
                result = trainer.run(data=gold_df)

        # Assert — boundary: >= threshold → promoted
        assert result["model_path"] is not None, (
            "model_path must not be None when lift == threshold (>= not >)"
        )
        assert os.path.exists(result["model_path"]), (
            f"model.pkl not found at {result['model_path']} on boundary lift"
        )

        run_tags = _extract_run_tags(mock_mlflow)
        assert run_tags.get("gate_result") == "promoted", (
            f"Expected gate_result='promoted' at boundary lift, "
            f"got: {run_tags.get('gate_result')}"
        )
