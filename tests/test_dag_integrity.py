"""DAG integrity harness.

PBI 25a established the directory baseline. PBI 25b adds
``_clock_factory.py`` (and later the DAG file itself). This harness
confirms the directory exists and contains only expected files.

Extension plan:
  - PBI 25b Leg 2+ adds TestRefresh15minDag: imports refresh_15min_dag,
    asserts dag_id == "refresh_15min_dag", task count, and schedule interval.
  - PBI 25c adds TestTrainMonthlyDag (dropped from build queue).
  - PBI 25d adds TestMonitorHourlyDag (dropped from build queue).

All DAG-structure assertions should live in this file to keep the DAG surface
area visible in one place.

AC #14.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# Files expected in orchestration/dags/ as of PBI 25b Leg 1.
_ALLOWED_PY_FILES = {"__init__.py", "_clock_factory.py"}


class TestDagsDirectoryBaseline:
    """Structural assertions on orchestration/dags/.

    Ensures only expected Python modules are present — catches accidental
    additions that bypass review.
    """

    def test_orchestration_dags_directory_exists_with_expected_files(
        self,
    ) -> None:
        """orchestration/dags/ exists and contains only allowed .py files."""
        # Arrange
        dags_dir = REPO_ROOT / "orchestration" / "dags"

        # Act
        py_files = {p.name for p in dags_dir.glob("*.py")}

        # Assert
        assert dags_dir.exists(), "orchestration/dags/ directory must exist"
        unexpected = py_files - _ALLOWED_PY_FILES
        assert not unexpected, (
            f"Unexpected .py files in orchestration/dags/: {unexpected}"
        )
