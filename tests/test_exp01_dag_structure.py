"""Structural assertions on the EXP-01 DAG wrapper.

The DAG file imports ``from airflow import DAG`` at module top. Airflow is NOT
installed in the host test env (only inside the experiment container), so these
tests SKIP when Airflow is absent — behaviour is verified in-container, while
structure is verified here whenever Airflow happens to be importable (e.g. CI
with the airflow extra). This mirrors how the project keeps DAG logic in a pure
module and asserts only the DAG surface here.

AC coverage (from PBI EXP-01 §6):
  - dag_id, schedule, catchup, max_active_runs
  - exactly 3 tasks: read_watermark -> advance_watermark -> write_watermark
  - linear dependency chain
"""

from __future__ import annotations

import pytest

pytest.importorskip("airflow", reason="Airflow only present inside the container")

from orchestration_experiment.dags.exp01_watermark_now_signal_dag import (  # noqa: E402
    dag,
)


class TestExp01DagStructure:
    """The EXP-01 DAG exposes the expected 3-task now-signal surface."""

    def test_dag_id(self) -> None:
        """dag_id matches the experiment naming convention."""
        # Arrange / Act / Assert
        assert dag.dag_id == "exp01_watermark_now_signal_dag"

    def test_schedule_is_15_min(self) -> None:
        """Cadence mirrors the production refresh interval."""
        # Arrange / Act / Assert
        assert dag.schedule_interval == "*/15 * * * *"

    def test_no_catchup_single_active_run(self) -> None:
        """Catchup off, serialized — no backfill storm, no overlap."""
        # Arrange / Act / Assert
        assert dag.catchup is False
        assert dag.max_active_runs == 1

    def test_exactly_three_tasks(self) -> None:
        """read_watermark, advance_watermark, write_watermark — nothing else."""
        # Arrange / Act
        task_ids = set(dag.task_dict)

        # Assert
        assert task_ids == {
            "read_watermark",
            "advance_watermark",
            "write_watermark",
        }

    def test_linear_dependency_chain(self) -> None:
        """read -> advance -> write."""
        # Arrange / Act
        read = dag.get_task("read_watermark")
        advance = dag.get_task("advance_watermark")
        write = dag.get_task("write_watermark")

        # Assert
        assert advance.upstream_task_ids == {"read_watermark"}
        assert write.upstream_task_ids == {"advance_watermark"}
        assert read.upstream_task_ids == set()
