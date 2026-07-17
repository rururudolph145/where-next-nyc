# Implementation Plan — PBI EXP-03a (Validate, Bronze Schema Gate)

> **Source PBI:** `orchestration_experiment/PBI_EXP03a_validate.md`
> **Depends on:** EXP-02 (cumulative-bronze ingest DAG — GREEN, verified 2026-07-14).
> **Stack:** Airflow **2.10.3** LocalExecutor, custom Dockerfile.
> **Created:** 2026-07-17

---

## Overview

Insert a `validate` task between `ingest` and `advance_watermark` so a DAG run
fails and holds the watermark whenever bronze fails Pandera schema checks; every
run writes a `validation_*.json` log under the isolated experiment root. Built
strictly RED → GREEN with the pure helpers (`build_validate_config`,
`run_validate`) unit-tested against mocked `pd.read_parquet`,
`validate_bronze_schema`, `write_validation_log`, and an injected fake `Clock`.

---

## Decisions locked (with the user, 2026-07-17)

1. **PBI split** — the original EXP-03 (`validate + feature`) was split into
   **EXP-03a (this PBI, validate)** and **EXP-03b (feature silver→gold)**. Each
   inserts exactly one task, one dep bump, one smoke cycle.
2. **OQ1 answer — inject config dict** (the seam EXP-02 uses). `run_validate`
   takes a `config` dict, not an explicit `log_dir`. EXP-03b will inherit the
   same seam.
3. **OQ2 answer — empty bronze passes** the validate gate. `rows_in=0` is a
   legitimate outcome; EXP-05's `check_no_new_data` handles that separately.
4. **Clock injection over `datetime` monkey-patching** — pass a fake `Clock`
   into `write_validation_log`, matching the pattern used by
   `app/ingestion/schema.py` (and the "no `time.sleep` in tests" testing rule).

---

## Key findings from investigation

- `app/ingestion/schema.py` already provides `validate_bronze_schema(df)`
  (raises `DataValidationError` with a `.failure_reasons` list) and
  `write_validation_log(log_dir, df, failure_reasons, *, clock=None)`
  (writes a `validation_<timestamp>.json`; uses `WallClock()` when `clock` is
  None). We reuse both verbatim — no new schema logic.
- `app/core/paths.py::validation_log_dir(config)` returns
  `Path(config["data"]["silver_path"]).parent / "_validation"`. So overriding
  only `data.silver_path` in the config dict is enough to redirect logs under
  the experiment root; no silver parquet needs to exist yet.
- `DataCleaner.run` (production) already does the exact `try/except → finally:
  write_validation_log` shape we need — we copy it into `run_validate`.
- `Ingestion.__init__` asserts `ingest.slice_duration == feature.bucket`; the
  validate override never touches those keys, so the guard survives.
- EXP-02's test file (`tests/test_exp02_ingest.py`) is the canonical pattern:
  module docstring listing AC coverage, class-grouped tests, AAA comments,
  `from __future__ import annotations`, `unittest.mock.patch` at the module
  boundary. EXP-03a mirrors it exactly.
- pandera is NOT in the baked EXP image (deliberately omitted in EXP-02's
  `requirements-exp.txt`). EXP-03a adds it.

---

## Affected Files

- `orchestration_experiment/dags/exp03a_validate.py` — **new** pure/Airflow-free
  helper: `build_validate_config`, `run_validate`.
- `orchestration_experiment/dags/exp03a_validate_dag.py` — **new** DAG wrapper.
  5-task linear chain, dual-context import (same pattern as EXP-02), `_validate`
  pulls `bronze_path` from `ingest`'s XCom.
- `orchestration_experiment/requirements-exp.txt` — **update**: add
  `pandera>=0.20`. Update the header comment to note EXP-03a adds it.
- `orchestration_experiment/Dockerfile` — no code change; rebuild picks it up.
- `orchestration_experiment/docker-compose-LocalExecutor.yml` — no change.
  The writable `./data:/opt/airflow/data_exp` mount already covers the
  `_validation/` subdir.
- `tests/test_exp03a_validate.py` — **new** RED-first unit tests (source below).
- `tests/test_exp03a_dag_structure.py` — **new** 5-task structural test with
  `pytest.importorskip("airflow")`; mirrors `test_exp02_dag_structure.py`.
- `orchestration_experiment/PBI_EXP03a_validate.md` — **update**: append
  verification note (OQ1/OQ2 answers, smoke results) after GREEN.

---

## The failing test (RED source — write this file first, verbatim)

Save as `tests/test_exp03a_validate.py`. Every test starts as a `ModuleNotFoundError`
until step 3 lands the helper.

```python
"""Tests for orchestration_experiment/dags/exp03a_validate.py.

PBI EXP-03a (experimentation) — the pure helpers the validate DAG delegates to.
Inserts a ``validate`` task between ``ingest`` and ``advance_watermark`` so
malformed bronze raises ``RuntimeError`` → ``advance``/``write`` skip under
``all_success`` → watermark held. Every run (pass OR fail) writes a
``validation_<timestamp>.json`` log under the experiment silver root.

The logic lives in a PURE, Airflow-free module (the experiment container has
Airflow; the host test env does NOT). Two seams keep it unit-testable:

  * ``build_validate_config`` — deep-overrides ``data.silver_path`` so
    ``validation_log_dir(config)`` resolves under the experiment root (never
    production ``data/``). Nothing else changes.
  * ``run_validate`` — reads bronze parquet, calls ``validate_bronze_schema``,
    writes a log via ``write_validation_log`` inside ``finally`` (so failures
    are logged too), re-raises ``DataValidationError`` as ``RuntimeError`` for
    Airflow.

AC coverage (from PLAN_EXP03a §Implementation Steps 1-3):
  - build_validate_config overrides data.silver_path
  - build_validate_config does not mutate the base config (deep copy)
  - build_validate_config preserves unrelated keys
  - run_validate reads bronze from the given path
  - run_validate calls validate_bronze_schema with the loaded DataFrame
  - run_validate writes a log with empty failure_reasons on the happy path
  - run_validate returns the bronze path on success (stable XCom flow)
  - run_validate re-raises DataValidationError as RuntimeError
  - run_validate writes a log with failure_reasons BEFORE re-raising
  - run_validate threads the injected Clock through to write_validation_log
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.ingestion.schema import DataValidationError
from orchestration_experiment.dags.exp03a_validate import (
    build_validate_config,
    run_validate,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

BRONZE_PATH = "/opt/airflow/data_exp/bronze/yellow_tripdata/bronze.parquet"
SILVER_ROOT = "/opt/airflow/data_exp/silver/yellow_tripdata_cleaned.parquet"


class _FrozenClock:
    """Minimal ``Clock``-shaped stub with a fixed ``now()``.

    Not a mock: ``write_validation_log`` only calls ``.now()``, so a real object
    with the right method is simpler and more honest than a MagicMock.
    """

    def now(self) -> datetime:
        return datetime(2021, 1, 18, 0, 15, 0, tzinfo=timezone.utc)


@pytest.fixture
def base_config() -> dict[str, Any]:
    """Minimal config mirroring the shape ``validation_log_dir`` reads."""
    return {
        "ingest": {
            "slice_duration": "15min",
            "bronze_schema": {"tpep_pickup_datetime": "datetime64[us]"},
        },
        "feature": {"bucket": "15min"},
        "data": {
            "raw_path": "data/raw/yellow_tripdata",
            "bronze_path": "data/bronze/yellow_tripdata/bronze.parquet",
            "silver_path": "data/silver/yellow_tripdata_cleaned.parquet",
        },
    }


@pytest.fixture
def override_config(base_config: dict[str, Any]) -> dict[str, Any]:
    """A config already pointed at the experiment silver root."""
    return build_validate_config(base_config, silver_root=SILVER_ROOT)


# ---------------------------------------------------------------------------
# TestBuildValidateConfig
# ---------------------------------------------------------------------------


class TestBuildValidateConfig:
    """build_validate_config injects the isolated silver path (the seam)."""

    def test_overrides_silver_path(self, base_config: dict[str, Any]) -> None:
        """data.silver_path points at the injected experiment silver root."""
        # Act
        result = build_validate_config(base_config, silver_root=SILVER_ROOT)

        # Assert
        assert result["data"]["silver_path"] == SILVER_ROOT

    def test_does_not_mutate_base(self, base_config: dict[str, Any]) -> None:
        """The base config is deep-copied, never mutated in place."""
        # Arrange
        snapshot = copy.deepcopy(base_config)

        # Act
        build_validate_config(base_config, silver_root=SILVER_ROOT)

        # Assert
        assert base_config == snapshot

    def test_preserves_unrelated_keys(self, base_config: dict[str, Any]) -> None:
        """Only data.silver_path changes; everything else passes through."""
        # Act
        result = build_validate_config(base_config, silver_root=SILVER_ROOT)

        # Assert
        assert result["data"]["raw_path"] == base_config["data"]["raw_path"]
        assert result["data"]["bronze_path"] == base_config["data"]["bronze_path"]
        assert result["ingest"] == base_config["ingest"]
        assert result["feature"] == base_config["feature"]


# ---------------------------------------------------------------------------
# TestRunValidateHappyPath
# ---------------------------------------------------------------------------


class TestRunValidateHappyPath:
    """run_validate loads bronze, validates it, and logs a clean pass."""

    def test_reads_bronze_from_given_path(
        self, override_config: dict[str, Any]
    ) -> None:
        """pd.read_parquet is called with the bronze_path arg unchanged."""
        # Arrange
        df = pd.DataFrame({"PULocationID": [1.0]})
        with (
            patch(
                "orchestration_experiment.dags.exp03a_validate.pd.read_parquet",
                return_value=df,
            ) as mock_read,
            patch(
                "orchestration_experiment.dags.exp03a_validate.validate_bronze_schema"
            ),
            patch(
                "orchestration_experiment.dags.exp03a_validate.write_validation_log"
            ),
        ):
            # Act
            run_validate(BRONZE_PATH, config=override_config)

        # Assert
        mock_read.assert_called_once_with(BRONZE_PATH)

    def test_calls_validate_bronze_schema_with_dataframe(
        self, override_config: dict[str, Any]
    ) -> None:
        """The DataFrame returned by read_parquet is what gets validated."""
        # Arrange
        df = pd.DataFrame({"PULocationID": [1.0]})
        with (
            patch(
                "orchestration_experiment.dags.exp03a_validate.pd.read_parquet",
                return_value=df,
            ),
            patch(
                "orchestration_experiment.dags.exp03a_validate.validate_bronze_schema"
            ) as mock_validate,
            patch(
                "orchestration_experiment.dags.exp03a_validate.write_validation_log"
            ),
        ):
            # Act
            run_validate(BRONZE_PATH, config=override_config)

        # Assert
        mock_validate.assert_called_once()
        (received_df,), _ = mock_validate.call_args
        assert received_df is df

    def test_writes_log_with_empty_failure_reasons(
        self, override_config: dict[str, Any]
    ) -> None:
        """Happy-path log carries failure_reasons=[] (rows_failing==0)."""
        # Arrange
        df = pd.DataFrame({"PULocationID": [1.0]})
        with (
            patch(
                "orchestration_experiment.dags.exp03a_validate.pd.read_parquet",
                return_value=df,
            ),
            patch(
                "orchestration_experiment.dags.exp03a_validate.validate_bronze_schema"
            ),
            patch(
                "orchestration_experiment.dags.exp03a_validate.write_validation_log"
            ) as mock_log,
        ):
            # Act
            run_validate(BRONZE_PATH, config=override_config)

        # Assert
        mock_log.assert_called_once()
        args, kwargs = mock_log.call_args
        # Positional args: (log_dir, df, failure_reasons)
        assert args[2] == []
        # log_dir resolves under the overridden silver root
        assert str(args[0]).endswith("/silver/_validation")

    def test_returns_bronze_path_on_success(
        self, override_config: dict[str, Any]
    ) -> None:
        """A stable XCom contract: run_validate hands the bronze path forward."""
        # Arrange
        df = pd.DataFrame({"PULocationID": [1.0]})
        with (
            patch(
                "orchestration_experiment.dags.exp03a_validate.pd.read_parquet",
                return_value=df,
            ),
            patch(
                "orchestration_experiment.dags.exp03a_validate.validate_bronze_schema"
            ),
            patch(
                "orchestration_experiment.dags.exp03a_validate.write_validation_log"
            ),
        ):
            # Act
            result = run_validate(BRONZE_PATH, config=override_config)

        # Assert
        assert result == BRONZE_PATH


# ---------------------------------------------------------------------------
# TestRunValidateFailurePath
# ---------------------------------------------------------------------------


class TestRunValidateFailurePath:
    """Malformed bronze → RuntimeError, but the log is written first."""

    def test_reraises_data_validation_error_as_runtime_error(
        self, override_config: dict[str, Any]
    ) -> None:
        """DataValidationError becomes RuntimeError so Airflow marks it failed."""
        # Arrange
        df = pd.DataFrame({"PULocationID": [999.0]})
        reasons = ["PULocationID out of range [1, 263]"]
        with (
            patch(
                "orchestration_experiment.dags.exp03a_validate.pd.read_parquet",
                return_value=df,
            ),
            patch(
                "orchestration_experiment.dags.exp03a_validate.validate_bronze_schema",
                side_effect=DataValidationError(reasons),
            ),
            patch(
                "orchestration_experiment.dags.exp03a_validate.write_validation_log"
            ),
        ):
            # Act / Assert
            with pytest.raises(RuntimeError) as exc_info:
                run_validate(BRONZE_PATH, config=override_config)

        assert "PULocationID out of range" in str(exc_info.value)

    def test_writes_log_with_failure_reasons_before_reraising(
        self, override_config: dict[str, Any]
    ) -> None:
        """The finally-block log write fires even when validate raises."""
        # Arrange
        df = pd.DataFrame({"PULocationID": [999.0]})
        reasons = ["PULocationID out of range [1, 263]"]
        with (
            patch(
                "orchestration_experiment.dags.exp03a_validate.pd.read_parquet",
                return_value=df,
            ),
            patch(
                "orchestration_experiment.dags.exp03a_validate.validate_bronze_schema",
                side_effect=DataValidationError(reasons),
            ),
            patch(
                "orchestration_experiment.dags.exp03a_validate.write_validation_log"
            ) as mock_log,
        ):
            # Act
            with pytest.raises(RuntimeError):
                run_validate(BRONZE_PATH, config=override_config)

        # Assert
        mock_log.assert_called_once()
        args, _ = mock_log.call_args
        assert args[2] == reasons  # failure_reasons positional arg

    def test_clock_passed_through_to_log_writer(
        self, override_config: dict[str, Any]
    ) -> None:
        """A caller-supplied Clock reaches write_validation_log unchanged."""
        # Arrange
        df = pd.DataFrame({"PULocationID": [1.0]})
        clock = _FrozenClock()
        with (
            patch(
                "orchestration_experiment.dags.exp03a_validate.pd.read_parquet",
                return_value=df,
            ),
            patch(
                "orchestration_experiment.dags.exp03a_validate.validate_bronze_schema"
            ),
            patch(
                "orchestration_experiment.dags.exp03a_validate.write_validation_log"
            ) as mock_log,
        ):
            # Act
            run_validate(BRONZE_PATH, config=override_config, clock=clock)

        # Assert
        _, kwargs = mock_log.call_args
        assert kwargs.get("clock") is clock
```

### Expected RED output

```
$ uv run pytest tests/test_exp03a_validate.py
...
ImportError while importing test module '.../tests/test_exp03a_validate.py'.
...
ModuleNotFoundError: No module named 'orchestration_experiment.dags.exp03a_validate'
```

If ANY test passes, or the ImportError is on something other than
`exp03a_validate`, stop — the RED is not proving what we think it is. Fix the
test file before writing production code.

---

## Implementation Steps

**Phase 1 — RED**

1. [ ] **Write `tests/test_exp03a_validate.py`** verbatim from the block above.
2. [ ] **Verify RED** — `uv run pytest tests/test_exp03a_validate.py`. Confirm
   the failure is `ModuleNotFoundError: No module named
   'orchestration_experiment.dags.exp03a_validate'` (not a typo, not an
   unrelated import). This is the "watched it fail for the right reason" gate.

**Phase 2 — GREEN (helper)**

3. [ ] **Implement `orchestration_experiment/dags/exp03a_validate.py`** — the
   minimum code to flip every RED test to GREEN. Shape:

   ```python
   """Pure validate helpers for the EXP-03a bronze-schema-gate DAG.

   Airflow-free by design: the experiment container has Airflow but the host
   test env does not, so the DAG's logic lives here where pytest can import it.
   The DAG file (``exp03a_validate_dag.py``) is a thin wrapper that pulls
   ``bronze_path`` from XCom, builds the override config, and calls
   ``run_validate``.

   Two seams keep this unit-testable:

     * :func:`build_validate_config` deep-overrides ``data.silver_path`` so
       ``validation_log_dir(config)`` resolves under the experiment root —
       never production ``data/``.
     * :func:`run_validate` reads bronze, calls ``validate_bronze_schema``,
       and writes a log inside ``finally`` (so failures are logged too). On
       ``DataValidationError`` it re-raises as ``RuntimeError`` so Airflow
       marks the task failed → ``advance``/``write`` skip under
       ``all_success`` → watermark held.
   """

   from __future__ import annotations

   import copy
   from typing import Any

   import pandas as pd

   from app.core.clock import Clock
   from app.core.paths import validation_log_dir
   from app.ingestion.schema import (
       DataValidationError,
       validate_bronze_schema,
       write_validation_log,
   )


   def build_validate_config(
       base: dict[str, Any], *, silver_root: str
   ) -> dict[str, Any]:
       """Return a copy of *base* with ``data.silver_path`` pointed at *silver_root*."""
       config = copy.deepcopy(base)
       config["data"]["silver_path"] = silver_root
       return config


   def run_validate(
       bronze_path: str,
       *,
       config: dict[str, Any],
       clock: Clock | None = None,
   ) -> str:
       """Validate bronze; write a log; re-raise on failure.

       Returns *bronze_path* unchanged on success so the DAG's XCom flow stays
       predictable (``ingest`` → ``validate`` → ``advance`` all pass the path
       forward without transformation).
       """
       df = pd.read_parquet(bronze_path)
       log_dir = validation_log_dir(config)
       failure_reasons: list[str] = []
       try:
           validate_bronze_schema(df)
       except DataValidationError as exc:
           failure_reasons = exc.failure_reasons
           raise RuntimeError(
               f"bronze validation failed: {failure_reasons}"
           ) from exc
       finally:
           write_validation_log(log_dir, df, failure_reasons, clock=clock)
       return bronze_path
   ```

4. [ ] **Verify GREEN** — `uv run pytest tests/test_exp03a_validate.py`. All
   tests pass; output pristine (no warnings). Then `uv run pytest` full suite:
   confirm baseline unchanged apart from the new tests.

**Phase 3 — RED then GREEN for DAG structure**

5. [ ] **Write `tests/test_exp03a_dag_structure.py`** — copy
   `tests/test_exp02_dag_structure.py` verbatim; change:
   - module docstring to reference EXP-03a and step 7 below;
   - the import: `from orchestration_experiment.dags.exp03a_validate_dag import dag`;
   - `assert dag.dag_id == "exp03a_validate_dag"`;
   - `test_exactly_five_tasks` — expected set:
     `{"read_watermark", "ingest", "validate", "advance_watermark", "write_watermark"}`;
   - `test_linear_dependency_chain` — `validate.upstream == {"ingest"}`,
     `advance.upstream == {"validate"}`;
   - keep `test_schedule_is_15_min`, `test_no_catchup_single_active_run`,
     `test_advance_and_write_require_all_success`.

6. [ ] **Verify RED** — `uv run pytest tests/test_exp03a_dag_structure.py`.
   On the host Airflow is absent, so `pytest.importorskip("airflow")` skips
   the file — that IS the RED here (module doesn't exist yet). Structural
   correctness is authoritatively verified in-container in step 12.

7. [ ] **Implement `orchestration_experiment/dags/exp03a_validate_dag.py`** —
   clone `exp02_ingest_bronze_dag.py`; then:
   - Add dual-context import for the new helper:
     ```python
     try:
         from orchestration_experiment.dags.exp03a_validate import (
             build_validate_config, run_validate,
         )
     except ModuleNotFoundError:
         from exp03a_validate import (  # type: ignore[no-redef,import-not-found]
             build_validate_config, run_validate,
         )
     ```
   - Add constant `SILVER_ROOT =
     "/opt/airflow/data_exp/silver/yellow_tripdata_cleaned.parquet"`.
   - Add `_validate` callable:
     ```python
     def _validate(**context: object) -> None:
         ti = context["ti"]  # type: ignore[index]
         bronze_path: str = ti.xcom_pull(task_ids="ingest")  # type: ignore[attr-defined]
         config = build_validate_config(load_config(), silver_root=SILVER_ROOT)
         run_validate(bronze_path, config=config)
         logger.info("validate: OK for %s", bronze_path)
     ```
   - Insert `validate = PythonOperator(task_id="validate",
     python_callable=_validate)` and rewire:
     `read_watermark >> ingest >> validate >> advance_watermark >> write_watermark`.
     `advance`/`write` keep `trigger_rule="all_success"`.
   - Change `dag_id` to `exp03a_validate_dag`.

8. [ ] **Verify GREEN** — `uv run pytest tests/test_exp03a_dag_structure.py`.
   On host: still skipped (Airflow absent). In-container: step 12 exercises it.

**Phase 4 — build, boot, smoke**

9. [ ] **Deps + rebuild** — add `pandera>=0.20` to
   `orchestration_experiment/requirements-exp.txt`; update the header comment
   ("EXP-03a introduces pandera for bronze validation"). Then
   `docker compose -f orchestration_experiment/docker-compose-LocalExecutor.yml
   build`. Note build-log size delta.

10. [ ] **Permissions reset** — same UID-50000 gotcha as EXP-02:
    `docker compose ... exec -u 0 airflow-scheduler chown -R 50000:0
    /opt/airflow/data_exp`. Reseed `data/watermark.json` to
    `2021-01-18T00:00:00` if it advanced during EXP-02 smoke.

11. [ ] **Boot + import-error check (AIRFLOW QUIRK)** — `docker compose ... up
    -d`. FIRST command: `airflow dags list-import-errors`. Fix anything it
    returns before touching the UI.

12. [ ] **Smoke test (happy path)** — `airflow dags test exp03a_validate_dag
    2021-01-18`:
    - watermark advances `2021-01-18T00:00 → 00:15`;
    - `data_exp/silver/_validation/validation_*.json` exists;
    - log's `rows_failing == 0`, `rows_in > 0`, `validated_at` is ISO-formatted.

13. [ ] **Smoke test (failure path — the AC only container smoke can prove)** —
    back up bronze: `cp bronze.parquet bronze.parquet.bak`. Corrupt via short
    in-container Python one-liner: read bronze, overwrite `PULocationID` with
    `999` on all rows, write back. Rerun `airflow dags test exp03a_validate_dag
    2021-01-19`:
    - `validate` state = `failed`;
    - `advance_watermark`, `write_watermark` state = `upstream_failed` (skipped);
    - `data/watermark.json` unchanged from step 12's position;
    - a new `validation_*.json` exists with `rows_failing > 0` and
      `failure_reasons` naming `PULocationID` out-of-range;
    - restore bronze: `mv bronze.parquet.bak bronze.parquet`.

**Phase 5 — REFACTOR + gates**

14. [ ] **REFACTOR** — read `exp03a_validate.py` once with fresh eyes. Likely
    nothing to do (~15 lines). If the docstring is missing the "log writes on
    failure too" contract, add one line. Rerun
    `uv run pytest tests/test_exp03a_validate.py` — must stay green.

15. [ ] **Boundary AC** — `grep -R "from orchestration"
    orchestration_experiment/dags/exp03a_* tests/test_exp03a_*` returns empty.
    `tests/test_pbi25a_orchestrator_skeleton.py::test_app_does_not_import_from_orchestration`
    still passes.

16. [ ] **Full quality gates** — `uv run ruff check .`, `uv run ruff format
    --check .`, `uv run mypy app tests`, `uv run pytest`. Baseline: 372 passed
    / 2 skipped on `dev_MVP`. New count = 372 + 10 (helper tests) + 6
    (structural, skipped on host) — adjust if the actual test count differs.

17. [ ] **Fill PBI verification note** — append to `PBI_EXP03a_validate.md`:
    - **OQ1** answer: **(a) inject config dict** — matches EXP-02, keeps
      EXP-03b consistent.
    - **OQ2** answer: **empty bronze passes** — schema check is structural
      only; `rows_in=0` is a legitimate outcome.
    - Both smoke results (log paths + row counts + exact `failure_reasons`).
    - Do NOT commit. Per the user preference locked in `MEMORY.md`, stop after
      GREEN for review.

---

## Risks & Considerations

- **RED-step OQ1 lock-in** — the failing test's call signature
  (`run_validate(bronze_path, *, config, clock=None)`) commits us to injecting
  a config dict, closing PBI OQ1 as (a). Reversing later means rewriting the
  tests. Accepted trade — (a) matches EXP-02 and EXP-03b will inherit the same
  seam.
- **`finally`-block log write is the crash-safety contract** — the failure
  path test asserts both the `RuntimeError` reraise AND that
  `write_validation_log` was called. Miss the log-side assertion and a future
  refactor that drops the `finally` stays green. Explicit in the RED test.
- **Clock injection over `datetime` monkey-patching** — `write_validation_log`
  accepts `clock=None` and defaults to `WallClock()`. Tests pass a
  `_FrozenClock` stub (a real object, not a MagicMock — the surface is one
  method). Follows the "no `time.sleep` in tests" testing rule.
- **pandera adds ~30–50 MB to the baked image** — one-time rebuild in step 9;
  cached thereafter. Pin `>=0.20` to match `pyproject.toml`.
- **Bronze corruption in step 13 is destructive** — always `cp
  bronze.parquet bronze.parquet.bak` first. Fallback if the backup is lost:
  rerun EXP-02 for the corrupted slice.
- **XCom coupling to EXP-02** — `_validate` pulls `bronze_path` from `ingest`'s
  XCom, which is EXP-02's `_ingest` return value. A change to that return
  value would silently break EXP-03a. Not a bug today; noted so the reviewer
  sees the coupling.
- **Container-only smoke for the crash-safety AC** — the watermark-held-on-
  failure AC cannot be unit-tested at the DAG level without booting Airflow.
  Unit tests prove `run_validate` raises correctly; step 13 proves Airflow
  honors `all_success`. Both required.
- **Blast radius** — confined to `orchestration_experiment/` + two new test
  files + one line in `requirements-exp.txt`. No change to `app/`,
  `orchestration/`, `config.yml`, or `data/`.

---

## Open questions to resolve during build (from PBI EXP-03a §8)

- **OQ1** → **(a) inject config dict** (locked above); test signature commits
  to this before implementation.
- **OQ2** → **empty bronze passes** the gate (locked above); no row-count
  check added. EXP-05's `check_no_new_data` will handle the empty-window
  boundary case.
