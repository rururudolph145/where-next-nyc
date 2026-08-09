"""Tests for orchestration/lib/training_helpers.py + train_monthly_dag.

PBI 25c — the pure helpers the training DAG delegates to, plus the DAG's
structural contract. The logic lives in a PURE, Airflow-free module (the host
test env has no Airflow; the container does). Seams keep it unit-testable
without a live MLflow or a real gold parquet beyond a tiny fixture.

AC coverage (host):
  - three exception classes are defined
  - resolve_watermarks: bootstrap fallback to seed_start; is_bootstrap flag;
    raises WatermarkNotFoundError when the refresh watermark is absent
  - check_window_ready: skip when inference behind; pass at exact 30d boundary;
    WatermarkInvariantError when refresh < last_training
  - compute_window: full-history window (seed_start, last_training_wm + 30d)
  - read_gold_windowed: filters `hour` to [start, end); EmptyTrainingWindowError
  - tag_promoted_version: writes the trained_at_watermark ISO tag
  - run_train: passes the FILTERED df to Trainer; derives gate_result from
    model_path; queries + tags the version on promote only; crash propagates
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

import training_helpers as th  # type: ignore[import-not-found]

SEED_START = "2021-01-18T00:00:00"
SEED_DT = datetime(2021, 1, 18, 0, 0, 0)
MODEL_NAME = "ny_taxi_demand_model"


def _write_wm(root: Path, dt: datetime) -> None:
    """Write a watermark control file the way app.core.watermark reads it."""
    from app.core.watermark import write_watermark

    write_watermark(root, dt)


@pytest.fixture
def base_config() -> dict[str, Any]:
    return {
        "ingest": {"seed_start": SEED_START},
        "data": {"gold_path": "data/gold/yellow_tripdata"},
        "mlflow": {"model_name": MODEL_NAME, "tracking_uri": "sqlite:///mlflow.db"},
        "train": {"gate": {"min_wape_lift_pct": 2.0}},
    }


class TestWatermarkPlumbing:
    """Pure helpers: exceptions, watermark resolution, window logic, gold read."""

    def test_exception_classes_are_defined(self) -> None:
        """The three fault-model exceptions subclass Exception."""
        # Arrange / Act / Assert
        assert issubclass(th.WatermarkNotFoundError, Exception)
        assert issubclass(th.WatermarkInvariantError, Exception)
        assert issubclass(th.EmptyTrainingWindowError, Exception)

    def test_bootstrap_falls_back_to_seed_start(
        self, tmp_path: Path, base_config: dict[str, Any]
    ) -> None:
        """No last_training file → last_training_wm resolves to seed_start."""
        # Arrange
        refresh_root = tmp_path / "refresh_15min"
        last_root = tmp_path / "last_training"
        _write_wm(refresh_root, SEED_DT + timedelta(days=30))

        # Act
        result = th.resolve_watermarks(base_config, refresh_root, last_root)

        # Assert
        assert result["last_training_wm"] == SEED_DT.isoformat()

    def test_bootstrap_flag_true_when_file_absent(
        self, tmp_path: Path, base_config: dict[str, Any]
    ) -> None:
        """is_bootstrap is True when the last_training watermark is absent."""
        # Arrange
        refresh_root = tmp_path / "refresh_15min"
        _write_wm(refresh_root, SEED_DT + timedelta(days=30))

        # Act
        result = th.resolve_watermarks(base_config, refresh_root, tmp_path / "lt")

        # Assert
        assert result["is_bootstrap"] is True

    def test_bootstrap_flag_false_when_file_present(
        self, tmp_path: Path, base_config: dict[str, Any]
    ) -> None:
        """is_bootstrap is False once a last_training watermark exists."""
        # Arrange
        refresh_root = tmp_path / "refresh_15min"
        last_root = tmp_path / "last_training"
        _write_wm(refresh_root, SEED_DT + timedelta(days=60))
        _write_wm(last_root, SEED_DT + timedelta(days=30))

        # Act
        result = th.resolve_watermarks(base_config, refresh_root, last_root)

        # Assert
        assert result["is_bootstrap"] is False
        assert result["last_training_wm"] == (SEED_DT + timedelta(days=30)).isoformat()

    def test_missing_refresh_watermark_raises(
        self, tmp_path: Path, base_config: dict[str, Any]
    ) -> None:
        """Absent refresh watermark is invalid: training-before-inference."""
        # Arrange / Act / Assert
        with pytest.raises(th.WatermarkNotFoundError):
            th.resolve_watermarks(base_config, tmp_path / "refresh", tmp_path / "lt")

    def test_check_window_ready_skips_when_inference_behind(self) -> None:
        """Inference <30d ahead of training → AirflowSkipException."""
        # Arrange
        pytest.importorskip("airflow")
        from airflow.exceptions import AirflowSkipException  # type: ignore[import-not-found]

        last = SEED_DT
        refresh = SEED_DT + timedelta(days=15)

        # Act / Assert
        with pytest.raises(AirflowSkipException):
            th.check_window_ready(refresh, last)

    def test_check_window_ready_passes_at_exact_boundary(self) -> None:
        """refresh == last + 30d passes (>= not >); returns None, no raise."""
        # Arrange
        last = SEED_DT
        refresh = SEED_DT + timedelta(days=30)

        # Act
        result = th.check_window_ready(refresh, last)

        # Assert
        assert result is None

    def test_check_window_ready_raises_when_refresh_behind_training(self) -> None:
        """refresh < last_training is impossible under normal op → fail loud."""
        # Arrange
        last = SEED_DT + timedelta(days=30)
        refresh = SEED_DT

        # Act / Assert (invariant branch precedes the Airflow-dependent skip)
        with pytest.raises(th.WatermarkInvariantError):
            th.check_window_ready(refresh, last)

    def test_compute_window_full_history(self) -> None:
        """window_start == seed_start; window_end == last_training_wm + 30d."""
        # Arrange
        last = SEED_DT + timedelta(days=30)

        # Act
        start, end = th.compute_window(SEED_DT, last)

        # Assert
        assert start == SEED_DT
        assert end == last + timedelta(days=30)

    def test_reads_gold_windowed(self, tmp_path: Path) -> None:
        """The `hour` filter keeps [start, end) rows only (upper exclusive)."""
        # Arrange
        gold = tmp_path / "hourly_demand_featured.parquet"
        df = pd.DataFrame(
            {
                "hour": [
                    SEED_DT - timedelta(hours=1),  # before
                    SEED_DT,  # inside (start inclusive)
                    SEED_DT + timedelta(days=10),  # inside
                    SEED_DT + timedelta(days=30),  # end (exclusive)
                ],
                "trip_count": [1, 2, 3, 4],
            }
        )
        df.to_parquet(gold)

        # Act
        out = th.read_gold_windowed(gold, SEED_DT, SEED_DT + timedelta(days=30))

        # Assert
        assert list(out["trip_count"]) == [2, 3]

    def test_empty_slice_raises_loudly(self, tmp_path: Path) -> None:
        """A window selecting no rows raises EmptyTrainingWindowError."""
        # Arrange
        gold = tmp_path / "hourly_demand_featured.parquet"
        pd.DataFrame({"hour": [SEED_DT], "trip_count": [1]}).to_parquet(gold)

        # Act / Assert
        with pytest.raises(th.EmptyTrainingWindowError):
            th.read_gold_windowed(
                gold, SEED_DT + timedelta(days=99), SEED_DT + timedelta(days=100)
            )

    def test_tag_promoted_version_sets_iso_tag(self) -> None:
        """tag_promoted_version writes trained_at_watermark = window_end.isoformat()."""
        # Arrange
        client = MagicMock()
        window_end = SEED_DT + timedelta(days=30)

        # Act
        th.tag_promoted_version(client, MODEL_NAME, "7", window_end)

        # Assert
        client.set_model_version_tag.assert_called_once_with(
            MODEL_NAME, "7", "trained_at_watermark", window_end.isoformat()
        )


class TestTrainCallable:
    """run_train wires read_gold_windowed → Trainer → gate derivation → tag."""

    def _gold(self, tmp_path: Path) -> Path:
        gold = tmp_path / "hourly_demand_featured.parquet"
        pd.DataFrame(
            {"hour": [SEED_DT + timedelta(days=5)], "trip_count": [3]}
        ).to_parquet(gold)
        return gold

    def _client(self) -> MagicMock:
        client = MagicMock()
        client.get_latest_versions.return_value = [SimpleNamespace(version="7")]
        return client

    def test_trainer_receives_filtered_dataframe(self, tmp_path: Path) -> None:
        """Trainer.run is called with data=<filtered df>, not data=None."""
        # Arrange
        gold = self._gold(tmp_path)
        fake_trainer = MagicMock()
        fake_trainer.run.return_value = {
            "model_path": "/x/model.pkl",
            "wape_lift_pct": 5.0,
        }

        # Act
        th.run_train(
            gold_file=gold,
            window_start=SEED_DT,
            window_end=SEED_DT + timedelta(days=30),
            model_name=MODEL_NAME,
            trainer_factory=lambda: fake_trainer,
            client_factory=self._client,
        )

        # Assert
        _, kwargs = fake_trainer.run.call_args
        assert isinstance(kwargs["data"], pd.DataFrame)
        assert list(kwargs["data"]["trip_count"]) == [3]

    def test_xcom_out_contains_gate_result_and_version(self, tmp_path: Path) -> None:
        """Promote → {gate_result: promoted, model_version: '7', wape_lift_pct}."""
        # Arrange
        gold = self._gold(tmp_path)
        fake_trainer = MagicMock()
        fake_trainer.run.return_value = {
            "model_path": "/x/model.pkl",
            "wape_lift_pct": 5.0,
        }

        # Act
        out = th.run_train(
            gold_file=gold,
            window_start=SEED_DT,
            window_end=SEED_DT + timedelta(days=30),
            model_name=MODEL_NAME,
            trainer_factory=lambda: fake_trainer,
            client_factory=self._client,
        )

        # Assert
        assert out == {
            "gate_result": "promoted",
            "model_version": "7",
            "wape_lift_pct": 5.0,
        }

    def test_trained_at_watermark_tag_written_on_promote(self, tmp_path: Path) -> None:
        """On promote the version is tagged with window_end.isoformat()."""
        # Arrange
        gold = self._gold(tmp_path)
        window_end = SEED_DT + timedelta(days=30)
        fake_trainer = MagicMock()
        fake_trainer.run.return_value = {
            "model_path": "/x/model.pkl",
            "wape_lift_pct": 5.0,
        }
        client = self._client()

        # Act
        th.run_train(
            gold_file=gold,
            window_start=SEED_DT,
            window_end=window_end,
            model_name=MODEL_NAME,
            trainer_factory=lambda: fake_trainer,
            client_factory=lambda: client,
        )

        # Assert
        client.set_model_version_tag.assert_called_once_with(
            MODEL_NAME, "7", "trained_at_watermark", window_end.isoformat()
        )

    def test_trained_at_watermark_tag_not_written_on_reject(
        self, tmp_path: Path
    ) -> None:
        """Reject (model_path None) → no tag, model_version None, gate rejected."""
        # Arrange
        gold = self._gold(tmp_path)
        fake_trainer = MagicMock()
        fake_trainer.run.return_value = {"model_path": None, "wape_lift_pct": 1.0}
        client = self._client()

        # Act
        out = th.run_train(
            gold_file=gold,
            window_start=SEED_DT,
            window_end=SEED_DT + timedelta(days=30),
            model_name=MODEL_NAME,
            trainer_factory=lambda: fake_trainer,
            client_factory=lambda: client,
        )

        # Assert
        assert out["gate_result"] == "rejected"
        assert out["model_version"] is None
        client.set_model_version_tag.assert_not_called()

    def test_trainer_crash_propagates(self, tmp_path: Path) -> None:
        """Trainer.run raising → run_train re-raises (crash→hold semantics)."""
        # Arrange
        gold = self._gold(tmp_path)
        fake_trainer = MagicMock()
        fake_trainer.run.side_effect = RuntimeError("boom")

        # Act / Assert
        with pytest.raises(RuntimeError):
            th.run_train(
                gold_file=gold,
                window_start=SEED_DT,
                window_end=SEED_DT + timedelta(days=30),
                model_name=MODEL_NAME,
                trainer_factory=lambda: fake_trainer,
                client_factory=self._client,
            )


class TestDagStructure:
    """DAG structural contract — runs in-container only (Airflow present)."""

    @pytest.fixture(autouse=True)
    def _require_airflow(self) -> None:
        pytest.importorskip("airflow", reason="Airflow only in container")

    @pytest.fixture
    def dag(self):  # type: ignore[no-untyped-def]
        from orchestration.dags.train_monthly_dag import dag as _dag

        return _dag

    def test_dag_parses_cleanly(self) -> None:
        """DagBag imports the DAG with no errors."""
        # Arrange / Act
        from airflow.models import DagBag  # type: ignore[import-not-found]

        bag = DagBag(dag_folder="orchestration/dags/", include_examples=False)

        # Assert
        assert bag.import_errors == {}

    def test_dag_id_is_train_monthly_dag(self, dag) -> None:  # type: ignore[no-untyped-def]
        # Arrange / Act / Assert
        assert dag.dag_id == "train_monthly_dag"

    def test_schedule_is_none(self, dag) -> None:  # type: ignore[no-untyped-def]
        # Arrange / Act / Assert
        assert dag.schedule_interval is None

    def test_max_active_runs_is_one(self, dag) -> None:  # type: ignore[no-untyped-def]
        # Arrange / Act / Assert
        assert dag.max_active_runs == 1

    def test_catchup_is_false(self, dag) -> None:  # type: ignore[no-untyped-def]
        # Arrange / Act / Assert
        assert dag.catchup is False

    def test_tags_contain_training_and_monthly(self, dag) -> None:  # type: ignore[no-untyped-def]
        # Arrange / Act / Assert
        assert "training" in dag.tags
        assert "monthly" in dag.tags

    def test_task_chain_is_linear(self, dag) -> None:  # type: ignore[no-untyped-def]
        """Six tasks in the expected linear order."""
        # Arrange
        expected_tasks = {
            "read_watermarks",
            "check_window_ready",
            "compute_window",
            "feature",
            "train",
            "write_training_watermark",
        }

        # Act
        actual_tasks = {t.task_id for t in dag.tasks}

        # Assert
        assert actual_tasks == expected_tasks

        task_map = {t.task_id: t for t in dag.tasks}
        assert task_map["read_watermarks"].downstream_task_ids == {"check_window_ready"}
        assert task_map["check_window_ready"].downstream_task_ids == {"compute_window"}
        assert task_map["compute_window"].downstream_task_ids == {"feature"}
        assert task_map["feature"].downstream_task_ids == {"train"}
        assert task_map["train"].downstream_task_ids == {"write_training_watermark"}
        assert task_map["write_training_watermark"].downstream_task_ids == set()


class TestTrainDagFeatureTask:
    """_feature task calls run_feature with for_training=True into a train-gold path."""

    def test_feature_callable_calls_run_feature_with_for_training_true(self) -> None:
        """_feature() passes for_training=True to run_feature."""
        # Arrange
        pytest.importorskip("airflow")
        from unittest.mock import patch, MagicMock

        from orchestration.dags.train_monthly_dag import _feature

        ti_mock = MagicMock()
        ti_mock.xcom_pull.return_value = (
            "/opt/airflow/data/bronze/yellow_tripdata/bronze.parquet"
        )
        context = {"ti": ti_mock}

        with patch(
            "orchestration.dags.train_monthly_dag.run_feature"
        ) as mock_run_feature:
            mock_run_feature.return_value = ti_mock.xcom_pull.return_value
            # Act
            _feature(**context)

        # Assert — for_training=True was passed
        _, kwargs = mock_run_feature.call_args
        assert kwargs.get("for_training") is True

    def test_feature_callable_uses_train_gold_path(self) -> None:
        """_feature() writes to a train-specific gold path, not inference gold."""
        # Arrange
        pytest.importorskip("airflow")
        from unittest.mock import patch, MagicMock

        from orchestration.dags.train_monthly_dag import _feature

        ti_mock = MagicMock()
        ti_mock.xcom_pull.return_value = (
            "/opt/airflow/data/bronze/yellow_tripdata/bronze.parquet"
        )
        context = {"ti": ti_mock}

        with patch(
            "orchestration.dags.train_monthly_dag.run_feature"
        ) as mock_run_feature:
            mock_run_feature.return_value = ti_mock.xcom_pull.return_value
            with patch(
                "orchestration.dags.train_monthly_dag.build_feature_config"
            ) as mock_bfc:
                mock_bfc.return_value = {}
                # Act
                _feature(**context)

        # Assert — gold_root passed to build_feature_config differs from inference gold
        _, kwargs = mock_bfc.call_args
        assert kwargs["gold_root"] != "/opt/airflow/data/gold/yellow_tripdata"
        assert "train" in kwargs["gold_root"]


class TestWindowDaysConfig:
    """Step 4 — window_days wired from config."""

    def test_compute_window_respects_window_days_param(self) -> None:
        """window_days=7 overrides the 30-day default in compute_window."""
        # Arrange
        from orchestration.lib import training_helpers as th

        seed = datetime(2021, 1, 18)
        last = seed + timedelta(days=30)

        # Act
        _, end = th.compute_window(seed, last, window_days=7)

        # Assert — window_end is last + 7d (not 30d)
        assert end == last + timedelta(days=7)

    def test_check_window_ready_passes_at_custom_window_boundary(self) -> None:
        """check_window_ready passes when delta == window_days (custom value)."""
        # Arrange
        from orchestration.lib import training_helpers as th

        last = datetime(2021, 2, 1)
        refresh = last + timedelta(days=7)

        # Act — should NOT raise (delta equals the window requirement)
        th.check_window_ready(refresh, last, window_days=7)

    def test_check_window_ready_skips_when_below_custom_window(self) -> None:
        """check_window_ready raises AirflowSkipException when delta < window_days."""
        # Arrange
        import pytest

        try:
            from airflow.exceptions import AirflowSkipException
        except ImportError:
            pytest.skip("Airflow not installed in host env")

        from orchestration.lib import training_helpers as th

        last = datetime(2021, 2, 1)
        refresh = last + timedelta(days=3)  # 3 < 7

        # Act / Assert
        with pytest.raises(AirflowSkipException):
            th.check_window_ready(refresh, last, window_days=7)


class TestBuildTrainConfig:
    """build_train_config overrides model.store_path without mutating base."""

    def test_build_train_config_overrides_store_path(self) -> None:
        """build_train_config sets model.store_path to the supplied model_root."""
        # Arrange
        from orchestration.lib.training_helpers import build_train_config

        base = {"model": {"store_path": "data/models"}, "train": {}}

        # Act
        result = build_train_config(base, model_root="/opt/airflow/data/models")

        # Assert
        assert result["model"]["store_path"] == "/opt/airflow/data/models"

    def test_build_train_config_does_not_mutate_base(self) -> None:
        """build_train_config deep-copies — base config is unchanged."""
        # Arrange
        from orchestration.lib.training_helpers import build_train_config

        base = {"model": {"store_path": "data/models"}, "train": {}}

        # Act
        build_train_config(base, model_root="/opt/airflow/data/models")

        # Assert
        assert base["model"]["store_path"] == "data/models"
