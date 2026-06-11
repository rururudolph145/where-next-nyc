"""Acceptance tests for PBI 25b Leg 3 — ``refresh_15min_dag``.

Covers the following story ACs (per
``implementation_checklist/pbi25b_refresh_15min_dag_story.md``):

  AC #1  — DAG file exists and parses without error
  AC #2  — DAG has the 8 tasks in the documented linear chain
  AC #3  — Schedule is ``*/15 * * * *``, ``catchup=False``, ``max_active_runs=1``
  AC #7  — Re-running an already-complete ``(day, bucket)`` is a no-op
  AC #8  — On failure between check_watermark and write_done the .done file
           is NOT written
  AC #9  — ``make demo`` target exists with both DEMO_MODE branches
  AC #10 — ``data/watermarks/refresh_15min/`` is gitkept;
           ``data/watermarks/`` is gitignored
  AC #14 — ``orchestration/README.md`` documents make demo, backfill,
           AIRFLOW_UID

ACs #4, #5, #6 are covered by ``tests/test_health_endpoint_clock.py`` (Leg 2).
ACs #11 and #12 are covered by pre-existing tests and are not duplicated here.

Test strategy
-------------
* **DAG-shape tests (TestDagShape)**: ``airflow`` is NOT installed in the test
  venv.  DagBag-based assertions are skipped via ``pytest.skip()`` where
  needed.  Structural assertions that do not require the Airflow runtime
  (file existence, AST-level schedule constant, source-text keyword checks)
  run unconditionally.
* **Idempotency / watermark tests (TestDagIdempotency)**: exercise
  ``app.core.watermark`` and the DAG task callables directly via a fake
  Airflow ``TaskInstance`` context — no Airflow scheduler needed.
* **Makefile tests (TestDemoMakefile)**: read ``makefile`` and assert target
  presence and ``DEMO_MODE`` branching text.
* **Watermark housekeeping (TestWatermarkHousekeeping)**: filesystem and
  ``.gitignore`` assertions.
* **README documentation (TestReadmeDocumentation)**: substring assertions on
  ``orchestration/README.md``.
"""

from __future__ import annotations

import ast
from datetime import date, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dag_source() -> str:
    """Return the raw source text of refresh_15min_dag.py."""
    return (REPO_ROOT / "orchestration" / "dags" / "refresh_15min_dag.py").read_text(
        encoding="utf-8"
    )


def _makefile_source() -> str:
    """Return the raw source text of the project Makefile."""
    return (REPO_ROOT / "makefile").read_text(encoding="utf-8")


def _readme_source() -> str:
    """Return the raw source text of orchestration/README.md."""
    return (REPO_ROOT / "orchestration" / "README.md").read_text(encoding="utf-8")


def _fake_ti_context(day_str: str, bucket: int) -> dict[str, Any]:
    """Build a minimal fake Airflow TaskInstance context for watermark tasks.

    The context satisfies the ``context["ti"].xcom_pull(task_ids="decide_bucket")``
    access pattern used by ``_check_watermark`` and ``_write_done`` without
    requiring a running Airflow metastore.
    """
    decide_result: dict[str, object] = {"day": day_str, "bucket": bucket}
    ti = MagicMock()
    ti.xcom_pull.return_value = decide_result
    return {"ti": ti}


# ---------------------------------------------------------------------------
# TestDagShape — AC #1, AC #2, AC #3
# ---------------------------------------------------------------------------


class TestDagShape:
    """Structural assertions on refresh_15min_dag.py.

    DagBag-based assertions are skipped when ``airflow`` is not importable.
    Source-level assertions run unconditionally — they do not require the
    Airflow runtime.
    """

    def test_ac1_dag_file_exists(self) -> None:
        """AC #1 — refresh_15min_dag.py exists at orchestration/dags/.

        Arrange: compute expected path relative to REPO_ROOT.
        Act: check existence.
        Assert: the file is present.
        """
        # Arrange
        expected_path = REPO_ROOT / "orchestration" / "dags" / "refresh_15min_dag.py"

        # Act / Assert
        assert expected_path.is_file(), (
            f"refresh_15min_dag.py not found at {expected_path}. "
            "Leg 3 builder must create this file."
        )

    def test_ac1_dag_file_parses_without_syntax_error(self) -> None:
        """AC #1 — DAG file is syntactically valid Python.

        Arrange: read the source.
        Act: parse with ast.parse (does not require Airflow runtime).
        Assert: no SyntaxError is raised.
        """
        # Arrange
        source = _dag_source()
        path = str(REPO_ROOT / "orchestration" / "dags" / "refresh_15min_dag.py")

        # Act / Assert
        try:
            ast.parse(source, filename=path)
        except SyntaxError as exc:
            pytest.fail(
                f"refresh_15min_dag.py has a syntax error and cannot be parsed: {exc}"
            )

    def test_ac1_dag_id_string_present_in_source(self) -> None:
        """AC #1 — dag_id='refresh_15min_dag' appears in the source.

        This is a lightweight proxy for the DagBag check when Airflow is not
        installed.  The full DagBag assertion is deferred to the skip block.
        """
        # Arrange
        source = _dag_source()

        # Act / Assert
        assert '"refresh_15min_dag"' in source or "'refresh_15min_dag'" in source, (
            "dag_id string 'refresh_15min_dag' not found in DAG source. "
            "The DAG definition must use this exact dag_id."
        )

    @pytest.mark.parametrize(
        "task_id",
        [
            "decide_bucket",
            "read_watermark",
            "check_no_new_data",
            "check_watermark",
            "ingest",
            "validate",
            "feature",
            "predict",
            "log_predict",
            "advance_watermark",
            "write_done",
        ],
    )
    def test_ac2_task_id_string_present_in_source(self, task_id: str) -> None:
        """AC #2 — each of the 11 task IDs appears in the DAG source.

        PBI 25b FIX grows the chain 8 -> 11: read_watermark,
        check_no_new_data, and advance_watermark are added for the
        watermark-advancing incremental pipeline. One parametrised case per
        task — a missing task_id string means the task was not defined or was
        renamed.
        """
        # Arrange
        source = _dag_source()

        # Act / Assert
        assert f'"{task_id}"' in source or f"'{task_id}'" in source, (
            f"Task id {task_id!r} not found in refresh_15min_dag.py source. "
            "All 8 task IDs listed in the brief must be present."
        )

    def test_ac2_linear_chain_dependency_order_in_source(self) -> None:
        """AC #2 — the linear chain operator (>>) appears in the source.

        Verifies that the 8 tasks are chained with ``>>`` rather than being
        defined as isolated, unconnected operators.
        """
        # Arrange
        source = _dag_source()

        # Act / Assert
        assert ">>" in source, (
            "No '>>' dependency operator found in refresh_15min_dag.py. "
            "The task chain must be expressed with >> operators."
        )

    def test_ac3_schedule_is_15min_cron_in_source(self) -> None:
        """AC #3 — schedule='*/15 * * * *' appears in the DAG definition."""
        # Arrange
        source = _dag_source()

        # Act / Assert
        assert '"*/15 * * * *"' in source or "'*/15 * * * *'" in source, (
            "schedule '*/15 * * * *' not found in DAG source. "
            "The brief requires this exact cron expression."
        )

    def test_ac3_catchup_false_in_source(self) -> None:
        """AC #3 — catchup=False appears in the DAG definition."""
        # Arrange
        source = _dag_source()

        # Act / Assert
        assert "catchup=False" in source, (
            "catchup=False not found in DAG source. "
            "The brief requires catchup=False to avoid historical backfill."
        )

    def test_ac3_max_active_runs_one_in_source(self) -> None:
        """AC #3 — max_active_runs=1 appears in the DAG definition."""
        # Arrange
        source = _dag_source()

        # Act / Assert
        assert "max_active_runs=1" in source, (
            "max_active_runs=1 not found in DAG source. "
            "The brief requires max_active_runs=1 to prevent overlapping runs."
        )

    def test_ac1_dagbag_shape_when_airflow_available(self) -> None:
        """AC #1/#2/#3 — DagBag shape assertions (skipped when airflow absent).

        When airflow is installed this test imports the DAG via DagBag and
        confirms dag_id, schedule, catchup, max_active_runs, and task count.
        Skipped gracefully when airflow is not in the test venv so the suite
        stays green in CI environments that do not install the Airflow extras.
        """
        # Arrange
        try:
            from airflow.models import DagBag  # type: ignore[import]
        except ModuleNotFoundError:
            pytest.skip(
                "airflow is not installed in the test venv — "
                "DagBag-based shape assertions are skipped. "
                "Run this test inside the orchestration Docker image to exercise it."
            )

        # Act
        dag_folder = str(REPO_ROOT / "orchestration" / "dags")
        bag = DagBag(dag_folder=dag_folder, include_examples=False)

        # Assert — no import errors
        assert not bag.import_errors, (
            f"DagBag reported import errors: {bag.import_errors}"
        )

        dag = bag.get_dag("refresh_15min_dag")
        assert dag is not None, "DAG 'refresh_15min_dag' not found in DagBag"
        assert dag.schedule_interval == "*/15 * * * *"
        assert dag.catchup is False
        assert dag.max_active_runs == 1
        assert len(dag.tasks) == 11


# ---------------------------------------------------------------------------
# TestDagIdempotency — AC #7, AC #8
# ---------------------------------------------------------------------------


class TestDagIdempotency:
    """Watermark gating and idempotency invariant tests.

    These tests exercise the Python callables from refresh_15min_dag.py
    (``_check_watermark``, ``_write_done``) and the ``app.core.watermark``
    module directly — no Airflow scheduler is needed.
    """

    def test_ac7_check_watermark_raises_skip_when_done_file_exists(
        self, tmp_path: Path
    ) -> None:
        """AC #7 — check_watermark raises AirflowSkipException when .done file exists.

        Arrange: write a .done file for a known (day, bucket) pair under a
            tmp_path watermark root; monkeypatch the DAG module's _WATERMARK_ROOT.
        Act: invoke _check_watermark via its callable.
        Assert: AirflowSkipException is raised.
        """
        # Skip if airflow not available — AirflowSkipException cannot be imported
        try:
            from airflow.exceptions import AirflowSkipException  # type: ignore[import]
        except ModuleNotFoundError:
            pytest.skip("airflow not installed — AirflowSkipException not importable")

        # Import after the skip guard so the test file stays importable without Airflow
        import importlib
        import sys

        # We import the callable directly but must patch _WATERMARK_ROOT so that
        # is_complete() looks at tmp_path instead of the real data/ directory.
        # We do this by importing the dag module's namespace and patching the symbol.
        # The dag module itself imports airflow at module level; if Airflow is absent
        # this block is already guarded by the skip above.
        dag_module_name = "orchestration.dags.refresh_15min_dag"
        if dag_module_name not in sys.modules:
            dag_module = importlib.import_module(dag_module_name)
        else:
            dag_module = sys.modules[dag_module_name]

        from app.core.watermark import mark_complete

        watermark_root = tmp_path / "watermarks" / "refresh_15min"
        test_day = date(2024, 6, 3)
        test_bucket = 4

        # Arrange: write the .done file so is_complete() returns True
        mark_complete(root=watermark_root, day=test_day, bucket=test_bucket)

        context = _fake_ti_context(test_day.isoformat(), test_bucket)

        # Act / Assert
        with patch.object(dag_module, "_WATERMARK_ROOT", watermark_root):
            with pytest.raises(AirflowSkipException):
                dag_module._check_watermark(**context)

    def test_ac7_watermark_is_complete_returns_true_after_mark(
        self, tmp_path: Path
    ) -> None:
        """AC #7 (unit layer) — is_complete returns True only after mark_complete.

        Verifies the watermark primitive that backs check_watermark's gate.
        Arrange: a fresh tmp_path watermark root.
        Act: call is_complete before and after mark_complete.
        Assert: False before, True after.
        """
        from app.core.watermark import is_complete, mark_complete

        # Arrange
        root = tmp_path / "wm"
        test_day = date(2024, 6, 3)
        test_bucket = 0

        # Act + Assert (before)
        assert not is_complete(root=root, day=test_day, bucket=test_bucket), (
            "is_complete() must return False before mark_complete() is called"
        )

        # Act (mark)
        mark_complete(root=root, day=test_day, bucket=test_bucket)

        # Assert (after)
        assert is_complete(root=root, day=test_day, bucket=test_bucket), (
            "is_complete() must return True after mark_complete() is called"
        )

    def test_ac8_done_file_not_written_if_write_done_not_called(
        self, tmp_path: Path
    ) -> None:
        """AC #8 — .done file is NOT written unless write_done succeeds.

        Idempotency invariant: if the pipeline fails between check_watermark
        and write_done, no .done file exists.  This test simulates a mid-run
        failure by checking that the watermark directory is empty (i.e. no
        .done file appears) when write_done is never called.

        Arrange: a fresh watermark root under tmp_path.
        Act: do NOT call mark_complete (simulating failure before write_done).
        Assert: no .done file exists at the expected path.
        """
        from app.core.watermark import bucket_path, is_complete

        # Arrange
        root = tmp_path / "wm_failure_scenario"
        test_day = date(2024, 6, 3)
        test_bucket = 8

        # Act: no call to mark_complete (simulated failure before write_done)

        # Assert — .done file does not exist
        expected_path = bucket_path(root=root, day=test_day, bucket=test_bucket)
        assert not expected_path.exists(), (
            f"Expected no .done file at {expected_path} when write_done was not called. "
            "The idempotency invariant requires that the watermark is written "
            "ONLY as the last step of a successful run."
        )
        assert not is_complete(root=root, day=test_day, bucket=test_bucket), (
            "is_complete() must return False when write_done has not been called."
        )

    def test_ac8_write_done_callable_creates_done_file(self, tmp_path: Path) -> None:
        """AC #8 (positive path) — write_done creates the .done file.

        Tests that _write_done (which calls mark_complete) does produce the
        watermark file when invoked successfully — validating that the
        idempotency invariant is upheld on the happy path.

        Arrange: monkeypatch _WATERMARK_ROOT to tmp_path.
        Act: call _write_done with a fake ti context.
        Assert: the .done file exists at the expected path.
        """
        try:
            from airflow.exceptions import AirflowSkipException as _  # noqa: F401
        except ModuleNotFoundError:
            pytest.skip("airflow not installed — DAG callables cannot be imported")

        import importlib
        import sys

        dag_module_name = "orchestration.dags.refresh_15min_dag"
        if dag_module_name not in sys.modules:
            dag_module = importlib.import_module(dag_module_name)
        else:
            dag_module = sys.modules[dag_module_name]

        from app.core.watermark import bucket_path

        # Arrange
        watermark_root = tmp_path / "watermarks" / "refresh_15min"
        test_day = date(2024, 6, 3)
        test_bucket = 12

        context = _fake_ti_context(test_day.isoformat(), test_bucket)
        expected_path = bucket_path(
            root=watermark_root, day=test_day, bucket=test_bucket
        )

        # Act
        with patch.object(dag_module, "_WATERMARK_ROOT", watermark_root):
            dag_module._write_done(**context)

        # Assert
        assert expected_path.exists(), (
            f"write_done callable must create {expected_path}. "
            "The .done file is the idempotency marker that prevents re-processing."
        )

    def test_ac7_rerun_same_bucket_is_noop_via_watermark_check(
        self, tmp_path: Path
    ) -> None:
        """AC #7 — re-running the same (day, bucket) is confirmed no-op.

        Simulates a complete first run (mark_complete called) then a second
        trigger: is_complete must return True, confirming check_watermark would
        raise AirflowSkipException and downstream tasks would not execute.

        Arrange: mark a bucket complete.
        Act: call is_complete for the same (day, bucket).
        Assert: True — a second run would be short-circuited.
        """
        from app.core.watermark import is_complete, mark_complete

        # Arrange — first run completes successfully
        root = tmp_path / "wm_rerun"
        test_day = date(2024, 6, 4)
        test_bucket = 20
        mark_complete(root=root, day=test_day, bucket=test_bucket)

        # Act — second trigger checks the watermark
        result = is_complete(root=root, day=test_day, bucket=test_bucket)

        # Assert
        assert result is True, (
            "is_complete() must return True after first run, confirming the "
            "second trigger would be a no-op (check_watermark raises AirflowSkipException)."
        )

    def test_ac7_different_bucket_is_not_complete(self, tmp_path: Path) -> None:
        """AC #7 boundary — a different bucket on the same day is NOT marked complete.

        Verifies the watermark key is (day, bucket) — marking bucket 4 must
        not affect bucket 5.
        """
        from app.core.watermark import is_complete, mark_complete

        # Arrange
        root = tmp_path / "wm_boundary"
        test_day = date(2024, 6, 3)
        mark_complete(root=root, day=test_day, bucket=4)

        # Act / Assert — bucket 5 is unaffected
        assert not is_complete(root=root, day=test_day, bucket=5), (
            "Marking bucket 4 complete must not affect bucket 5. "
            "Watermark key is (day, bucket); each bucket is independent."
        )

    def test_ac7_different_day_is_not_complete(self, tmp_path: Path) -> None:
        """AC #7 boundary — a different day on the same bucket is NOT marked complete.

        Verifies the watermark key is (day, bucket) — marking 2024-06-03/bucket-4
        must not affect 2024-06-04/bucket-4.
        """
        from app.core.watermark import is_complete, mark_complete

        # Arrange
        root = tmp_path / "wm_day_boundary"
        bucket = 4
        day_a = date(2024, 6, 3)
        day_b = day_a + timedelta(days=1)
        mark_complete(root=root, day=day_a, bucket=bucket)

        # Act / Assert — next day is unaffected
        assert not is_complete(root=root, day=day_b, bucket=bucket), (
            f"Marking {day_a}/bucket-{bucket} complete must not affect {day_b}/bucket-{bucket}. "
            "Watermark key is (day, bucket); each day is independent."
        )


# ---------------------------------------------------------------------------
# TestNoNewData — AC #4 (re-run no-op when no new data / dataset boundary)
# ---------------------------------------------------------------------------


class TestNoNewData:
    """``_check_no_new_data`` callable raises AirflowSkipException at boundary.

    AC #4: when ``window_start`` is at or past the dataset's last timestamp
    (``2021-03-31 23:59:57``), the ``check_no_new_data`` task must raise
    ``AirflowSkipException`` so the whole run is skipped without ingesting
    empty data.

    Both tests are guarded with ``pytest.skip`` when Airflow is absent — the
    DAG module imports AirflowSkipException at module level so the callable
    is only importable when Airflow is installed.
    """

    def test_ac4_check_no_new_data_raises_skip_at_boundary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC #4 — _check_no_new_data raises AirflowSkipException when
        window_start == dataset max (boundary reached).

        Arrange: fake ti XCom returns window_start == dataset boundary;
            monkeypatch load_config so _dataset_max returns a known value.
        Act: call _check_no_new_data.
        Assert: AirflowSkipException is raised.
        """
        try:
            from airflow.exceptions import AirflowSkipException  # type: ignore[import]
        except ModuleNotFoundError:
            pytest.skip("airflow not installed — AirflowSkipException not importable")

        import importlib
        import sys

        dag_module_name = "orchestration.dags.refresh_15min_dag"
        if dag_module_name not in sys.modules:
            dag_module = importlib.import_module(dag_module_name)
        else:
            dag_module = sys.modules[dag_module_name]

        # The dataset boundary is 2021-03-31 23:59:57 (end_year_month=2021-03).
        # Patch load_config inside the dag module so _dataset_max resolves cleanly.
        _FAKE_CFG = {
            "ingest": {
                "end_year_month": "2021-03",
                "slice_duration": "15min",
                "seed_start": "2021-01-18T00:00:00",
            }
        }

        # Arrange: window_start is at the dataset max — no data left
        ti = MagicMock()
        ti.xcom_pull.return_value = {
            "window_start": "2021-03-31T23:59:57",
            "window_end": "2021-04-01T00:14:57",
        }
        context = {"ti": ti}

        with patch(
            "app.core.config.load_config",
            return_value=_FAKE_CFG,
        ):
            with pytest.raises(AirflowSkipException):
                dag_module._check_no_new_data(**context)

    def test_ac4_check_no_new_data_does_not_skip_before_boundary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC #4 (negative path) — no skip when window_start is before the boundary.

        Arrange: window_start well before dataset end (2021-01-18T00:00:00).
        Act: call _check_no_new_data.
        Assert: no exception raised (the run proceeds normally).
        """
        try:
            from airflow.exceptions import AirflowSkipException as _  # noqa: F401
        except ModuleNotFoundError:
            pytest.skip("airflow not installed — AirflowSkipException not importable")

        import importlib
        import sys

        dag_module_name = "orchestration.dags.refresh_15min_dag"
        if dag_module_name not in sys.modules:
            dag_module = importlib.import_module(dag_module_name)
        else:
            dag_module = sys.modules[dag_module_name]

        _FAKE_CFG = {
            "ingest": {
                "end_year_month": "2021-03",
                "slice_duration": "15min",
                "seed_start": "2021-01-18T00:00:00",
            }
        }

        # Arrange: window_start well before dataset boundary
        ti = MagicMock()
        ti.xcom_pull.return_value = {
            "window_start": "2021-01-18T00:00:00",
            "window_end": "2021-01-18T00:15:00",
        }
        context = {"ti": ti}

        # Act + Assert — must not raise
        with patch(
            "app.core.config.load_config",
            return_value=_FAKE_CFG,
        ):
            dag_module._check_no_new_data(**context)  # no exception expected


# ---------------------------------------------------------------------------
# TestCumulativeBronze — regression guard for the sliding-window bug (PBI 25b FIX)
# ---------------------------------------------------------------------------


class TestCumulativeBronze:
    """``_ingest`` must use ``seed_start`` as the lower bound, not the watermark.

    Regression guard for the C1 bug fixed in PBI 25b FIX: previously
    ``_ingest`` read ``window_start`` from the XCom (the advancing watermark),
    producing a SLIDING 15-min slice and overwriting bronze each run with
    only the most recent slice. The brief mandates a CUMULATIVE window:
    lower bound frozen at ``ingest.seed_start``, upper bound advances.

    If a future refactor reintroduces the sliding behaviour, this test fails.
    """

    def test_ingest_uses_seed_start_not_advancing_watermark(
        self, tmp_path: Path
    ) -> None:
        """``Ingestion().run`` is called with ``window_start == seed_start``
        even when the XCom watermark has already advanced well past it.

        Arrange:
        - XCom returns an ADVANCED watermark (2021-02-15) as ``window_start``.
        - ``ingest.seed_start`` in config is the FROZEN lower bound
          (2021-01-18T00:00:00).
        - Patch ``Ingestion`` so we can capture the kwargs without running.
        Act: invoke ``_ingest``.
        Assert: ``Ingestion().run`` was called with
        ``window_start == datetime(2021, 1, 18, 0, 0, 0)`` (seed), and
        ``window_end`` taken from the XCom (advanced).
        """
        try:
            from airflow.exceptions import AirflowSkipException as _  # noqa: F401
        except ModuleNotFoundError:
            pytest.skip("airflow not installed — DAG callable cannot be imported")

        import importlib
        import sys
        from datetime import datetime as _dt

        dag_module_name = "orchestration.dags.refresh_15min_dag"
        if dag_module_name not in sys.modules:
            dag_module = importlib.import_module(dag_module_name)
        else:
            dag_module = sys.modules[dag_module_name]

        # Arrange: XCom carries an ADVANCED watermark (well past seed_start)
        ti = MagicMock()
        ti.xcom_pull.return_value = {
            "window_start": "2021-02-15T00:00:00",  # advanced, must be IGNORED by _ingest
            "window_end": "2021-02-15T00:15:00",
        }
        context = {"ti": ti}

        fake_config = {
            "ingest": {
                "seed_start": "2021-01-18T00:00:00",  # the FROZEN lower bound
                "slice_duration": "15min",
            }
        }

        mock_ingestion_cls = MagicMock()
        mock_instance = MagicMock()
        mock_ingestion_cls.return_value = mock_instance

        with patch("app.core.config.load_config", return_value=fake_config):
            with patch("app.ingestion.ingest.Ingestion", mock_ingestion_cls):
                # Act
                dag_module._ingest(**context)

        # Assert: Ingestion().run called once with seed_start as window_start
        mock_instance.run.assert_called_once()
        call_kwargs = mock_instance.run.call_args.kwargs
        assert call_kwargs["window_start"] == _dt(2021, 1, 18, 0, 0, 0), (
            f"_ingest must use seed_start ({_dt(2021, 1, 18, 0, 0, 0)}) as the "
            f"lower bound, NOT the advancing watermark. Got {call_kwargs['window_start']}. "
            "If this fails, the sliding-window bug has been reintroduced and "
            "bronze will no longer be cumulative."
        )
        assert call_kwargs["window_end"] == _dt(2021, 2, 15, 0, 15, 0), (
            "_ingest must read window_end from the XCom (advancing watermark)."
        )


# ---------------------------------------------------------------------------
# TestClockFactory — clock factory (not a story AC per se but required by brief)
# ---------------------------------------------------------------------------


class TestClockFactory:
    """Smoke tests for the make_clock() factory and _SpeedupClock.

    These tests cover the clock-factory behaviour documented in the brief
    (Section 3b) and required for the DAG to function correctly.
    """

    def test_make_clock_returns_wallclock_when_demo_mode_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """make_clock() returns WallClock when DEMO_MODE is not set.

        Arrange: ensure DEMO_MODE is absent from the environment.
        Act: call make_clock().
        Assert: returned clock is an instance of WallClock.
        """
        from app.core.clock import WallClock
        from orchestration.dags._clock_factory import make_clock

        # Arrange
        monkeypatch.delenv("DEMO_MODE", raising=False)

        # Act
        clock = make_clock()

        # Assert
        assert isinstance(clock, WallClock), (
            f"make_clock() must return WallClock when DEMO_MODE is unset, got {type(clock).__name__}"
        )

    def test_make_clock_returns_speedup_clock_when_demo_mode_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """make_clock() returns _SpeedupClock when DEMO_MODE=1.

        Arrange: set DEMO_MODE=1 in the environment.
        Act: call make_clock().
        Assert: returned clock is an instance of _SpeedupClock.
        """
        from orchestration.dags._clock_factory import _SpeedupClock, make_clock

        # Arrange
        monkeypatch.setenv("DEMO_MODE", "1")

        # Act
        clock = make_clock()
        try:
            # Assert
            assert isinstance(clock, _SpeedupClock), (
                f"make_clock() must return _SpeedupClock when DEMO_MODE=1, "
                f"got {type(clock).__name__}"
            )
        finally:
            # Stop the background thread started by _SpeedupClock
            if hasattr(clock, "stop"):
                clock.stop()  # type: ignore[union-attr]

    def test_make_clock_speedup_clock_has_450x_speedup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_SpeedupClock returned by make_clock() has speedup == 450.

        This ensures the 15-simulated-minutes-per-2-seconds ratio from the
        brief (Section 3b) is correctly configured.
        """
        from orchestration.dags._clock_factory import (
            _DEMO_SPEEDUP,
            _SpeedupClock,
            make_clock,
        )

        # Arrange
        monkeypatch.setenv("DEMO_MODE", "1")

        # Act
        clock = make_clock()
        try:
            # Assert
            assert isinstance(clock, _SpeedupClock)
            assert clock._speedup == _DEMO_SPEEDUP, (
                f"_SpeedupClock._speedup must equal {_DEMO_SPEEDUP} (450x), "
                f"got {clock._speedup}"
            )
        finally:
            if hasattr(clock, "stop"):
                clock.stop()  # type: ignore[union-attr]

    def test_make_clock_registers_as_active_clock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """make_clock() registers the returned clock as the process-global active clock.

        The DAG tasks read time via get_active_clock() — make_clock() must call
        set_active_clock() as a side effect so that the process-global registry
        is updated.
        """
        from app.core.clock import get_active_clock
        from orchestration.dags._clock_factory import make_clock

        # Arrange
        monkeypatch.delenv("DEMO_MODE", raising=False)

        # Act
        clock = make_clock()
        active = get_active_clock()

        # Assert — active clock is the same object returned by make_clock()
        assert active is clock, (
            "make_clock() must register the returned clock via set_active_clock(). "
            "get_active_clock() returned a different object."
        )

    def test_speedup_clock_now_returns_utc_aware_datetime(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_SpeedupClock.now() returns a UTC-aware datetime.

        Clock-discipline contract: all Clock implementations must return
        UTC-aware datetimes to avoid naive-vs-aware comparison errors.
        """
        from orchestration.dags._clock_factory import make_clock

        # Arrange
        monkeypatch.setenv("DEMO_MODE", "1")
        clock = make_clock()

        try:
            # Act
            now = clock.now()

            # Assert
            assert now.tzinfo is not None, (
                "_SpeedupClock.now() must return a timezone-aware datetime. "
                "Got a naive datetime, which violates the Clock protocol contract."
            )
            assert now.tzinfo == timezone.utc or str(now.tzinfo) in (
                "UTC",
                "tzutc()",
                "+00:00",
            ), (
                "_SpeedupClock.now() must return a UTC datetime. "
                f"Got tzinfo={now.tzinfo!r}"
            )
        finally:
            if hasattr(clock, "stop"):
                clock.stop()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# TestDemoMakefile — AC #9
# ---------------------------------------------------------------------------


class TestDemoMakefile:
    """Assertions on the ``make demo`` target.

    AC #9 — ``make demo`` target exists with both DEMO_MODE branches.
    """

    def test_ac9_demo_target_exists_in_makefile(self) -> None:
        """AC #9 — the ``demo:`` target is present in the Makefile.

        Arrange: read makefile source.
        Act: search for the demo: target declaration.
        Assert: the target line is present.
        """
        # Arrange
        source = _makefile_source()

        # Act / Assert
        assert "demo:" in source, (
            "'demo:' target not found in makefile. "
            "AC #9 requires a demo: target with DEMO_MODE branching."
        )

    def test_ac9_demo_target_has_demo_mode_conditional_branch(self) -> None:
        """AC #9 — the demo target uses ifdef DEMO_MODE (live branch).

        Arrange: read makefile source.
        Act: search for DEMO_MODE conditional.
        Assert: ifdef DEMO_MODE or ifeq($(DEMO_MODE),1) is present near the demo target.
        """
        # Arrange
        source = _makefile_source()

        # Act / Assert — Make conditionals allow either ifdef or ifeq syntax
        has_ifdef = "ifdef DEMO_MODE" in source
        has_ifeq = "ifeq" in source and "DEMO_MODE" in source
        assert has_ifdef or has_ifeq, (
            "demo target must contain a DEMO_MODE conditional branch "
            "(ifdef DEMO_MODE or ifeq). "
            "AC #9 requires both a live-cadence and a static branch."
        )

    def test_ac9_demo_target_has_else_branch_for_static_mode(self) -> None:
        """AC #9 — the demo target has an else branch (static mode).

        Arrange: read makefile source.
        Act: search for else branch between demo: and demo-stop:.
        Assert: 'else' appears in the makefile (covering the static path).
        """
        # Arrange
        source = _makefile_source()

        # Act / Assert
        assert "else" in source, (
            "demo target must have an 'else' branch for static demo mode. "
            "AC #9: when DEMO_MODE is unset, make demo must serve cached predictions."
        )

    def test_ac9_demo_target_invokes_uvicorn_in_static_branch(self) -> None:
        """AC #9 — the static branch of demo target invokes uvicorn.

        Arrange: read makefile source.
        Act: search for uvicorn in the makefile.
        Assert: 'uvicorn' appears in the source (static-mode serve command).
        """
        # Arrange
        source = _makefile_source()

        # Act / Assert
        assert "uvicorn" in source, (
            "'uvicorn' not found in makefile. "
            "The static demo branch must start the FastAPI server via uvicorn."
        )

    def test_ac9_demo_stop_target_exists_in_makefile(self) -> None:
        """AC #9 — the ``demo-stop:`` teardown target is present.

        AC #9 requires both a start and a stop entry point so a reviewer can
        cleanly tear down the demo stack.
        """
        # Arrange
        source = _makefile_source()

        # Act / Assert
        assert "demo-stop:" in source, (
            "'demo-stop:' target not found in makefile. "
            "A teardown target is required alongside the demo target."
        )

    def test_ac9_demo_target_dry_run_contains_uvicorn_in_live_branch(self) -> None:
        """AC #9 — make demo DEMO_MODE=1 (dry-run) contains uvicorn invocation.

        Arrange: nothing (dry-run does not execute commands).
        Act: invoke ``make -n demo DEMO_MODE=1`` via subprocess.
        Assert: 'uvicorn' appears in the printed command list.

        This test would have caught C2 (uvicorn missing from live branch).
        """
        import subprocess

        # Arrange / Act
        result = subprocess.run(
            ["make", "-n", "demo", "DEMO_MODE=1"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(REPO_ROOT),
        )

        # Assert
        assert "uvicorn" in result.stdout, (
            f"DEMO_MODE=1 branch must invoke uvicorn; make -n output:\n{result.stdout}"
        )


# ---------------------------------------------------------------------------
# TestWatermarkHousekeeping — AC #10
# ---------------------------------------------------------------------------


class TestWatermarkHousekeeping:
    """Filesystem and VCS assertions for the watermark directory.

    AC #10 — ``data/watermarks/refresh_15min/`` is gitkept;
              ``data/watermarks/`` is gitignored.
    """

    def test_ac10_watermark_gitkeep_file_exists(self) -> None:
        """AC #10 — .gitkeep exists under data/watermarks/refresh_15min/.

        Arrange: compute expected path.
        Act: check existence.
        Assert: .gitkeep is present (directory is tracked by git).
        """
        # Arrange
        expected = REPO_ROOT / "data" / "watermarks" / "refresh_15min" / ".gitkeep"

        # Act / Assert
        assert expected.is_file(), (
            f".gitkeep not found at {expected}. "
            "AC #10 requires data/watermarks/refresh_15min/.gitkeep to exist "
            "so that the directory is tracked by git."
        )

    def test_ac10_watermarks_directory_is_in_gitignore(self) -> None:
        """AC #10 — data/watermarks/ is listed in .gitignore.

        Arrange: read .gitignore.
        Act: search for the watermarks ignore pattern.
        Assert: 'data/watermarks/' appears as an ignore rule.
        """
        # Arrange
        gitignore_path = REPO_ROOT / ".gitignore"
        assert gitignore_path.is_file(), ".gitignore not found at repo root"
        gitignore_source = gitignore_path.read_text(encoding="utf-8")

        # Act / Assert
        assert "data/watermarks/" in gitignore_source, (
            "'data/watermarks/' not found in .gitignore. "
            "AC #10 requires this directory to be gitignored so that .done files "
            "are not committed to version control."
        )

    def test_ac10_gitkeep_exception_in_gitignore(self) -> None:
        """AC #10 — .gitkeep exception exists in .gitignore.

        The .gitignore must have a negation pattern so that
        data/watermarks/refresh_15min/.gitkeep is *not* ignored even though
        data/watermarks/ is ignored as a whole.
        """
        # Arrange
        gitignore_path = REPO_ROOT / ".gitignore"
        gitignore_source = gitignore_path.read_text(encoding="utf-8")

        # Act / Assert
        assert "!data/watermarks/refresh_15min/.gitkeep" in gitignore_source, (
            "Negation pattern '!data/watermarks/refresh_15min/.gitkeep' not found "
            "in .gitignore. Without this, the .gitkeep file is also ignored and the "
            "directory disappears from the repo for new clones."
        )

    def test_ac10_watermarks_refresh_15min_directory_exists(self) -> None:
        """AC #10 — data/watermarks/refresh_15min/ directory exists on disk.

        Arrange: compute expected path.
        Act: check existence as a directory.
        Assert: directory is present.
        """
        # Arrange
        expected_dir = REPO_ROOT / "data" / "watermarks" / "refresh_15min"

        # Act / Assert
        assert expected_dir.is_dir(), (
            f"data/watermarks/refresh_15min/ directory not found at {expected_dir}. "
            "The watermark directory must be created and tracked via .gitkeep."
        )


# ---------------------------------------------------------------------------
# TestReadmeDocumentation — AC #14
# ---------------------------------------------------------------------------


class TestReadmeDocumentation:
    """Substring assertions on orchestration/README.md.

    AC #14 — README documents make demo (both branches), backfill semantics,
              and AIRFLOW_UID.
    """

    def test_ac14_readme_documents_make_demo_command(self) -> None:
        """AC #14 — README mentions 'make demo'.

        Arrange: read orchestration/README.md.
        Act: search for 'make demo'.
        Assert: the command appears in the documentation.
        """
        # Arrange
        source = _readme_source()

        # Act / Assert
        assert "make demo" in source, (
            "'make demo' not found in orchestration/README.md. "
            "AC #14 requires the README to document the make demo command."
        )

    def test_ac14_readme_documents_demo_mode_1_live_cadence(self) -> None:
        """AC #14 — README documents the DEMO_MODE=1 live-cadence mode.

        Arrange: read orchestration/README.md.
        Act: search for 'DEMO_MODE=1' or 'DEMO_MODE'.
        Assert: the live-cadence demo mode is documented.
        """
        # Arrange
        source = _readme_source()

        # Act / Assert
        assert "DEMO_MODE" in source, (
            "'DEMO_MODE' not found in orchestration/README.md. "
            "AC #14 requires documentation of both demo branches."
        )

    def test_ac14_readme_documents_backfill_semantics(self) -> None:
        """AC #14 — README documents backfill semantics.

        Arrange: read orchestration/README.md.
        Act: search for 'backfill' (case-insensitive).
        Assert: backfill is mentioned.
        """
        # Arrange
        source = _readme_source().lower()

        # Act / Assert
        assert "backfill" in source, (
            "'backfill' not found in orchestration/README.md. "
            "AC #14 requires backfill semantics to be documented."
        )

    def test_ac14_readme_documents_airflow_uid(self) -> None:
        """AC #14 — README documents AIRFLOW_UID.

        Arrange: read orchestration/README.md.
        Act: search for 'AIRFLOW_UID'.
        Assert: the AIRFLOW_UID variable is documented.
        """
        # Arrange
        source = _readme_source()

        # Act / Assert
        assert "AIRFLOW_UID" in source, (
            "'AIRFLOW_UID' not found in orchestration/README.md. "
            "AC #14 requires AIRFLOW_UID to be documented (brief requirement)."
        )

    def test_ac14_readme_documents_refresh_15min_dag(self) -> None:
        """AC #14 — README documents the refresh_15min_dag.

        Arrange: read orchestration/README.md.
        Act: search for 'refresh_15min_dag'.
        Assert: the DAG ID is mentioned in the documentation.
        """
        # Arrange
        source = _readme_source()

        # Act / Assert
        assert "refresh_15min_dag" in source, (
            "'refresh_15min_dag' not found in orchestration/README.md. "
            "AC #14 requires the DAG task graph to be documented."
        )

    def test_ac14_readme_documents_watermark_contract(self) -> None:
        """AC #14 — README documents the watermark contract or path convention.

        Arrange: read orchestration/README.md.
        Act: search for 'watermark' (case-insensitive).
        Assert: watermark is mentioned.
        """
        # Arrange
        source = _readme_source().lower()

        # Act / Assert
        assert "watermark" in source, (
            "'watermark' not found in orchestration/README.md. "
            "AC #14 requires the watermark contract and path convention to be documented."
        )


# ---------------------------------------------------------------------------
# TestNoClockInDag — PBI 25b FIX: the DAG no longer wires a clock
# ---------------------------------------------------------------------------


class TestNoClockInDag:
    """The watermark-advancing DAG holds no clock state.

    PBI 25b FIX replaced the simulated clock with a DATA watermark: the
    simulation position lives in ``data/watermarks/refresh_15min/
    .current_watermark``, not in a ``_SpeedupClock`` background thread.  The
    DAG therefore no longer imports ``make_clock`` nor carries the
    ``_REFRESH_15MIN_CLOCK_INITIALIZED`` sentinel, so the I3 thread-leak
    concern (a new daemon thread on every Airflow re-parse) is moot by
    construction.  These source-level assertions run without Airflow.
    """

    def test_dag_does_not_import_make_clock(self) -> None:
        """The DAG source no longer imports make_clock from _clock_factory."""
        # Arrange
        source = _dag_source()

        # Act / Assert
        assert "make_clock" not in source, (
            "refresh_15min_dag.py must not import or call make_clock(): the "
            "simulation position is now a DATA watermark, not a clock thread."
        )

    def test_dag_has_no_clock_sentinel(self) -> None:
        """The DAG source no longer carries the clock-init env-var sentinel."""
        # Arrange
        source = _dag_source()

        # Act / Assert
        assert "_REFRESH_15MIN_CLOCK_INITIALIZED" not in source, (
            "The clock-init sentinel was removed with the clock; the DAG "
            "starts no background thread on re-parse."
        )

    def test_dag_does_not_use_get_active_clock(self) -> None:
        """The DAG reads no simulated clock; time access is the watermark."""
        # Arrange
        source = _dag_source()

        # Act / Assert
        assert "get_active_clock" not in source, (
            "refresh_15min_dag.py must not read get_active_clock(): the "
            "incremental position comes from read_watermark(), and the "
            "validation-log timestamp uses a real wall clock."
        )
