"""Structural assertions on the EXP-03b feature DAG wrapper.

The DAG file imports ``from airflow import DAG`` at module top. Airflow is NOT
installed in the host test env (only inside the experiment container), so these
tests SKIP when Airflow is absent — behaviour is verified in-container (step 12
of PLAN_EXP03b), while structure is verified here whenever Airflow happens to
be importable. Mirrors tests/test_exp03a_dag_structure.py.

EXP-03b inserts a single ``feature`` task into the EXP-03a topology, BETWEEN
``validate`` and ``advance_watermark`` — running DataCleaner (bronze→silver)
then FeatureEngineer (silver→gold) under one label. The 6-task chain proves
that the watermark only advances after all transforms succeed.

AC coverage (from PLAN_EXP03b §Implementation Step 5):
  - dag_id, 15-min schedule, catchup off, single active run
  - exactly 6 tasks: read -> ingest -> validate -> feature -> advance -> write
  - linear dependency chain with feature between validate and advance
  - advance/write keep trigger_rule="all_success" (crash safety)
"""

from __future__ import annotations

import pytest

pytest.importorskip("airflow", reason="Airflow only present inside the container")

from orchestration_experiment.dags.exp03b_feature_dag import (  # noqa: E402
    dag,
)


class TestExp03bDagStructure:
    """The EXP-03b DAG exposes the expected 6-task feature surface."""

    def test_dag_id(self) -> None:
        """dag_id matches the experiment naming convention."""
        # Arrange / Act / Assert
        assert dag.dag_id == "exp03b_feature_dag"

    def test_schedule_is_15_min(self) -> None:
        """Cadence mirrors the production refresh interval."""
        # Arrange / Act / Assert
        assert dag.schedule_interval == "*/15 * * * *"

    def test_no_catchup_single_active_run(self) -> None:
        """Catchup off, serialized — no backfill storm, no overlap."""
        # Arrange / Act / Assert
        assert dag.catchup is False
        assert dag.max_active_runs == 1

    def test_exactly_six_tasks(self) -> None:
        """read, ingest, validate, feature, advance, write — nothing else."""
        # Arrange / Act
        task_ids = set(dag.task_dict)

        # Assert
        assert task_ids == {
            "read_watermark",
            "ingest",
            "validate",
            "feature",
            "advance_watermark",
            "write_watermark",
        }

    def test_linear_dependency_chain(self) -> None:
        """read -> ingest -> validate -> feature -> advance -> write."""
        # Arrange / Act
        read = dag.get_task("read_watermark")
        ingest = dag.get_task("ingest")
        validate = dag.get_task("validate")
        feature = dag.get_task("feature")
        advance = dag.get_task("advance_watermark")
        write = dag.get_task("write_watermark")

        # Assert
        assert read.upstream_task_ids == set()
        assert ingest.upstream_task_ids == {"read_watermark"}
        assert validate.upstream_task_ids == {"ingest"}
        assert feature.upstream_task_ids == {"validate"}
        assert advance.upstream_task_ids == {"feature"}
        assert write.upstream_task_ids == {"advance_watermark"}

    def test_advance_and_write_require_all_success(self) -> None:
        """Crash safety: the watermark advances only if feature succeeded."""
        # Arrange / Act
        advance = dag.get_task("advance_watermark")
        write = dag.get_task("write_watermark")

        # Assert
        assert advance.trigger_rule == "all_success"
        assert write.trigger_rule == "all_success"
