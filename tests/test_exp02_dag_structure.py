"""Structural assertions on the EXP-02 ingest DAG wrapper.

The DAG file imports ``from airflow import DAG`` at module top. Airflow is NOT
installed in the host test env (only inside the experiment container), so these
tests SKIP when Airflow is absent — behaviour is verified in-container, while
structure is verified here whenever Airflow happens to be importable (e.g. CI
with the airflow extra). Mirrors tests/test_exp01_dag_structure.py.

EXP-02 inserts a single ``ingest`` task into the EXP-01 topology, BETWEEN
``read_watermark`` and ``advance_watermark`` — proving the watermark advances
only after a successful cumulative-bronze write.

AC coverage (from PLAN_EXP02 §Implementation Step 4):
  - dag_id, 15-min schedule, catchup off, single active run
  - exactly 4 tasks: read -> ingest -> advance -> write
  - linear dependency chain with ingest between read and advance
  - advance/write keep trigger_rule="all_success" (crash safety)
"""

from __future__ import annotations

import pytest

pytest.importorskip("airflow", reason="Airflow only present inside the container")

from orchestration_experiment.dags.exp02_ingest_bronze_dag import (  # noqa: E402
    dag,
)


class TestExp02DagStructure:
    """The EXP-02 DAG exposes the expected 4-task ingest surface."""

    def test_dag_id(self) -> None:
        """dag_id matches the experiment naming convention."""
        # Arrange / Act / Assert
        assert dag.dag_id == "exp02_ingest_bronze_dag"

    def test_schedule_is_15_min(self) -> None:
        """Cadence mirrors the production refresh interval."""
        # Arrange / Act / Assert
        assert dag.schedule_interval == "*/15 * * * *"

    def test_no_catchup_single_active_run(self) -> None:
        """Catchup off, serialized — no backfill storm, no overlap."""
        # Arrange / Act / Assert
        assert dag.catchup is False
        assert dag.max_active_runs == 1

    def test_exactly_four_tasks(self) -> None:
        """read, ingest, advance, write — nothing else."""
        # Arrange / Act
        task_ids = set(dag.task_dict)

        # Assert
        assert task_ids == {
            "read_watermark",
            "ingest",
            "advance_watermark",
            "write_watermark",
        }

    def test_linear_dependency_chain(self) -> None:
        """read -> ingest -> advance -> write (ingest is the new middle task)."""
        # Arrange / Act
        read = dag.get_task("read_watermark")
        ingest = dag.get_task("ingest")
        advance = dag.get_task("advance_watermark")
        write = dag.get_task("write_watermark")

        # Assert
        assert read.upstream_task_ids == set()
        assert ingest.upstream_task_ids == {"read_watermark"}
        assert advance.upstream_task_ids == {"ingest"}
        assert write.upstream_task_ids == {"advance_watermark"}

    def test_advance_and_write_require_all_success(self) -> None:
        """Crash safety: the watermark advances only if ingest succeeded."""
        # Arrange / Act
        advance = dag.get_task("advance_watermark")
        write = dag.get_task("write_watermark")

        # Assert
        assert advance.trigger_rule == "all_success"
        assert write.trigger_rule == "all_success"
