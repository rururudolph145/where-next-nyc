"""Wiring-only structural test for 15_minute_refresh_inference_dag.

Skips when Airflow is not installed (host-only environment). Explicitly does
NOT assert an exact task count (per docs/superpowers/specs/2026-07-24-
orchestration-promotion-design.md §7.2 and the exp-dag-promotion-notes
memory). If a future PBI legitimately adds a task, only the wiring assertions
need updating.
"""

from __future__ import annotations

import pytest

pytest.importorskip("airflow", reason="Airflow only present inside the container")

from orchestration.dags.the_15_minute_refresh_inference_dag import dag  # noqa: E402


class TestRefreshInferenceDagStructure:
    """The promoted DAG exposes the expected 8-task inference surface."""

    def test_dag_id(self) -> None:
        """dag_id matches the production name locked in the design brainstorm."""
        # Arrange / Act / Assert
        assert dag.dag_id == "15_minute_refresh_inference_dag"

    def test_schedule_is_15_min(self) -> None:
        """Cadence is the production 15-minute cron."""
        # Arrange / Act / Assert
        assert dag.schedule_interval == "*/15 * * * *"

    def test_no_catchup_single_active_run(self) -> None:
        """Catchup off, serialized — no backfill storm, no overlap."""
        # Arrange / Act / Assert
        assert dag.catchup is False
        assert dag.max_active_runs == 1

    def test_task_set(self) -> None:
        """The 8 task_ids are exactly the promoted-topology set."""
        # Arrange / Act
        task_ids = set(dag.task_dict)

        # Assert
        assert task_ids == {
            "read_watermark",
            "check_no_new_data",
            "ingest",
            "validate",
            "feature",
            "predict",
            "log_predict",
            "write_watermark",
        }

    def test_check_no_new_data_between_read_and_ingest(self) -> None:
        """check_no_new_data sits immediately after read_watermark, before ingest."""
        # Arrange / Act
        check = dag.get_task("check_no_new_data")

        # Assert
        assert check.upstream_task_ids == {"read_watermark"}
        assert check.downstream_task_ids == {"ingest"}

    def test_linear_dependency_chain(self) -> None:
        """read -> check -> ingest -> validate -> feature -> predict -> log -> write."""
        # Arrange / Act
        read = dag.get_task("read_watermark")
        check = dag.get_task("check_no_new_data")
        ingest = dag.get_task("ingest")
        validate = dag.get_task("validate")
        feature = dag.get_task("feature")
        predict = dag.get_task("predict")
        log_predict = dag.get_task("log_predict")
        write = dag.get_task("write_watermark")

        # Assert
        assert read.upstream_task_ids == set()
        assert check.upstream_task_ids == {"read_watermark"}
        assert ingest.upstream_task_ids == {"check_no_new_data"}
        assert validate.upstream_task_ids == {"ingest"}
        assert feature.upstream_task_ids == {"validate"}
        assert predict.upstream_task_ids == {"feature"}
        assert log_predict.upstream_task_ids == {"predict"}
        assert write.upstream_task_ids == {"log_predict"}

    def test_write_watermark_uses_all_success(self) -> None:
        """Failure or skip must NOT advance the watermark."""
        # Arrange / Act
        write = dag.get_task("write_watermark")

        # Assert
        assert write.trigger_rule == "all_success"
