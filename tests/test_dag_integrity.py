"""DAG integrity harness — vacuously passing in PBI 25a.

In this PBI the orchestration/dags/ directory contains only a .gitkeep file;
no DAG Python modules exist yet. This harness confirms the directory exists
and is empty of .py files.

Extension plan:
  - PBI 25b adds TestRefresh15minDag: imports refresh_15min_dag, asserts
    dag_id == "refresh_15min_dag", task count, and schedule interval.
  - PBI 25c adds TestTrainMonthlyDag: imports train_monthly_dag, asserts
    dag_id == "train_monthly_dag", task count, and schedule interval.
  - PBI 25d adds TestMonitorHourlyDag: imports monitor_hourly_dag, asserts
    dag_id == "monitor_hourly_dag", task count, and schedule interval.

All DAG-structure assertions should live in this file to keep the DAG surface
area visible in one place.

AC #14.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


class TestDagsDirectoryBaseline:
    """Vacuous assertions that pass before any DAG files are present.

    Each DAG PBI (25b / 25c / 25d) extends this file with a new test class
    that imports the real DAG object and asserts its dag_id, task count, and
    schedule interval.
    """

    def test_orchestration_dags_directory_imports_cleanly(self) -> None:
        """orchestration/dags/ exists and contains no .py DAG files in PBI 25a.

        This test is vacuously passing by design — no DAGs ship in this PBI.
        """
        # Arrange
        dags_dir = REPO_ROOT / "orchestration" / "dags"

        # Act
        py_files = list(dags_dir.glob("*.py"))

        # Assert
        assert dags_dir.exists(), "orchestration/dags/ directory must exist"
        assert py_files == [], (
            f"Expected no .py files in orchestration/dags/ in PBI 25a, "
            f"found: {py_files}"
        )
