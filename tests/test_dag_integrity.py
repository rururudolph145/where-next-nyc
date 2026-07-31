"""DAG integrity harness.

PBI 25a established the directory baseline. PBI 25b adds
``_clock_factory.py`` (and later the DAG file itself). This harness
confirms the directory exists and contains only expected files.

Extension plan:
  - PBI 25b Leg 2+ adds TestRefresh15minDag: imports refresh_15min_dag,
    asserts dag_id == "refresh_15min_dag", task count, and schedule interval.
  - PBI 25c adds ``train_monthly_dag.py`` (shipped); TestTrainMonthlyDag lives in test_train_monthly_dag.py.
  - PBI 25d adds TestMonitorHourlyDag (dropped from build queue).

All DAG-structure assertions should live in this file to keep the DAG surface
area visible in one place.

AC #14.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# Files expected in orchestration/dags/ as of PBI 25c.
_ALLOWED_PY_FILES = {
    "__init__.py",
    "the_15_minute_refresh_inference_dag.py",
    "train_monthly_dag.py",
}

# Paths required for `make demo-tick` to succeed end-to-end.
_MODEL_PKL = REPO_ROOT / "data" / "models_trained" / "model.pkl"
_IQR_BOUNDS = REPO_ROOT / "data" / "models" / "iqr_bounds.json"


class TestDemoTickPreconditions:
    """Artifacts that must exist on the host before `make demo-tick` can succeed.

    Covers the three blockers identified in
    docs/superpowers/plans/2026-07-26-fix-ingest-permission-and-smoke-test.md:
    the 17-feature model.pkl and iqr_bounds.json. These are host-only artefacts
    (untracked by git) — if they are absent, the ingest/feature/predict tasks
    will fail inside the container regardless of the UID fix.
    """

    def test_model_pkl_exists(self) -> None:
        """data/models_trained/model.pkl must be present for the predict task."""
        # Arrange / Act / Assert
        assert _MODEL_PKL.exists(), (
            f"Missing {_MODEL_PKL}. "
            "Copy from mlruns/1/models/m-9a3a980ff2264b9b924f5424792b6117/"
            "artifacts/model.pkl"
        )

    def test_model_pkl_is_valid_joblib(self) -> None:
        """model.pkl must start with the pickle/joblib magic bytes (\\x80 protocol)."""
        # Arrange
        if not _MODEL_PKL.exists():
            import pytest

            pytest.skip("model.pkl absent — covered by test_model_pkl_exists")

        # Act
        header = _MODEL_PKL.read_bytes()[:2]

        # Assert — joblib uses pickle protocol 2+; first byte is 0x80, second is
        # the protocol number (2, 4, or 5 depending on Python version).
        assert header[0] == 0x80, (
            f"model.pkl does not look like a pickle file (header: {header!r})"
        )

    def test_iqr_bounds_exists(self) -> None:
        """data/models/iqr_bounds.json must be present for the feature task."""
        # Arrange / Act / Assert
        assert _IQR_BOUNDS.exists(), (
            f"Missing {_IQR_BOUNDS}. "
            "Copy from orchestration_experiment/data/models/iqr_bounds.json"
        )

    def test_iqr_bounds_has_required_keys(self) -> None:
        """iqr_bounds.json must contain q1, q3, iqr keys at the top level."""
        # Arrange
        if not _IQR_BOUNDS.exists():
            import pytest

            pytest.skip("iqr_bounds.json absent — covered by test_iqr_bounds_exists")

        import json

        # Act
        data = json.loads(_IQR_BOUNDS.read_text())

        # Assert
        missing = {"q1", "q3", "iqr"} - set(data.keys())
        assert not missing, (
            f"iqr_bounds.json missing keys: {missing}. Got: {list(data.keys())}"
        )


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
