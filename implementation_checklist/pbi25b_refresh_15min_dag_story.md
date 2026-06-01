# PBI 25b — `refresh_15min_dag`: User Story

## Why This Matters

PBI 25a established the `Clock` protocol and the `orchestration/` container scaffold. The portfolio still looks static: every demo is a single `curl` against a predictions file last written whenever someone last ran the pipeline manually. A hiring reviewer spending five minutes in the repo cannot tell whether the candidate has thought past "I trained a model" into "this thing has to run repeatedly, on a schedule, without leaking real-time assumptions into business logic."

PBI 25b closes that gap. It ships the first real Airflow DAG — a 15-minute ingestion-to-inference refresh cycle — and a `DEMO_MODE=1 make demo` entry point that lets the reviewer watch `bucket_time` advance on live `curl` calls. No new ML capability, no new model, no new endpoints. The deliverable is **operational realism under time**.

The intended audience for the demo is a portfolio reviewer (SmartNews / Google L4 MLE hiring manager). There is no live driver or production deployment in scope.

---

## User Story

As a portfolio reviewer evaluating this project, I want to run a single command that brings up the full stack in demo mode and shows predictions refreshing on a schedule, so that I can confirm the system is operationally coherent under time — not just a model wrapped in a one-shot script.

When this PBI ships, a reviewer who runs `DEMO_MODE=1 make demo` and then issues repeated `curl /heatmap` calls sees `bucket_time` advancing by 15 simulated minutes roughly every two seconds of wall-clock, exercising the same ingestion → feature → predict → log path that a production deployment would.

---

## Acceptance Criteria

### Demo entry point

1. After `cp orchestration/.env.example orchestration/.env`, running `DEMO_MODE=1 make demo` brings up FastAPI, the Airflow scheduler, and the Airflow webserver without any further manual steps. All processes start within a reasonable timeout (documented in `orchestration/README.md`).

2. Under `DEMO_MODE=1`, repeated `curl /heatmap` calls at ~2-second intervals show `bucket_time` advancing by exactly 15 simulated minutes per cycle. The same pipeline code path (ingest → validate → feature → predict → log) executes on each cycle.

3. When `DEMO_MODE` is unset, `make demo` starts FastAPI serving existing cached predictions without bringing up Airflow; `bucket_time` does not change between calls.

### `/health` clock transparency

4. When `DEMO_MODE` is unset, `GET /health` returns a payload containing `"simulated": false`. The fields `simulated_now` and `wall_clock_now` are absent from the payload.

5. When `DEMO_MODE=1`, `GET /health` returns a payload containing `"simulated": true`, an ISO-8601 UTC string `simulated_now` showing the current simulated time, and an ISO-8601 UTC string `wall_clock_now` showing actual wall-clock time. The two times are visibly different.

6. Two `/health` calls separated by approximately 2 seconds of wall-clock time under `DEMO_MODE=1` show `simulated_now` advancing by approximately 15 minutes (within a ±10% tolerance).

### DAG structure

7. `orchestration/dags/refresh_15min_dag.py` exists, imports without error, and defines a DAG with `dag_id="refresh_15min_dag"`, `schedule="*/15 * * * *"`, `catchup=False`, and `max_active_runs=1`.

8. The DAG contains exactly 8 tasks with IDs: `decide_bucket`, `check_watermark`, `ingest`, `validate`, `feature`, `predict`, `log_predict`, `write_done`, in the dependency order described in the brief (section 3a).

### Watermark idempotency

9. When a `.done` file already exists at `data/watermarks/refresh_15min/{date}/{bucket}.done`, a DAG run for that `(date, bucket)` short-circuits at `check_watermark` — the `ingest`, `validate`, `feature`, `predict`, `log_predict`, and `write_done` tasks do not execute.

10. A successful DAG run creates the watermark file at `data/watermarks/refresh_15min/{date}/{bucket}.done`. Re-triggering the same `(date, bucket)` thereafter is a confirmed no-op (no re-ingestion, no re-prediction, no duplicate log entry).

### Clock-discipline regression guard

11. A test in `tests/test_clock_discipline.py` performs a repo-wide grep over `app/` (excluding `app/core/clock.py`) and asserts that no production source file contains a bare `datetime.utcnow`, an unqualified `datetime.now(` without `tz=`, or a direct `time.sleep` call. This test is green after the 5-callsite refactor.

12. The dependency-direction invariant from PBI 25a continues to hold: no module under `app/` imports from `orchestration/`. The existing test `tests/test_pbi25a_orchestrator_skeleton.py::test_app_does_not_import_from_orchestration` remains green without modification.

### Quality gates

13. `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy app tests`, and `uv run pytest` all exit with code 0 on the PR branch.

### Documentation and tracker

14. `orchestration/README.md` is updated to document: the `make demo` command (both branches), the `refresh_15min_dag` task graph, the watermark contract and path convention, backfill semantics, and the scheduler heartbeat tuning (or the external-loop fallback if that is the chosen resolution to the scheduler-heartbeat open question — see Open Questions below).

15. `implementation_checklist/pbi_tracker.md` marks PBI 25b as shipped.

---

## Edge Cases

- **Naive `datetime` constructed inside a DAG task.** A naive datetime compared to a `SimulatedClock`-produced UTC-aware datetime raises a `TypeError` at runtime. The clock-discipline guard (AC #11) and `mypy` checks (AC #13) are the two lines of defence.

- **Two DAG runs racing on the same `(date, bucket)`.** `max_active_runs=1` at the DAG level prevents a second run from starting while the first is in-flight. The idempotent watermark (AC #9) is the fallback if that constraint is somehow bypassed.

- **`DEMO_MODE=1` set but `orchestration/.env` absent.** `make demo` must detect the missing file and fail loudly with a clear error message, not silently bring up a partial stack with missing variables.

- **Watermark file present but Gold parquet absent** (e.g. manual deletion). The DAG will see the `.done` file, treat the bucket as complete, and skip re-processing. This is intentional: a missing Gold file after a confirmed run is a manual-intervention case. The README must document this explicitly.

- **`data/watermarks/` directory absent on first run.** The `write_done` task must create the directory tree with `parents=True, exist_ok=True` before writing the `.done` file.

- **`_SpeedupClock.advance()` called concurrently from the background tick thread and a DAG task thread.** `_SpeedupClock` must guard both `advance()` and `now()` with a `threading.Lock`. `SimulatedClock` itself (in `app/core/`) remains single-threaded; the lock lives in `_SpeedupClock` in `orchestration/`.

- **`make demo` brought up a second time without `make airflow-down` first.** The target must handle an already-running Airflow stack gracefully (idempotent `docker compose up` semantics) rather than exiting non-zero.

---

## Out of Scope

- No new ML model, no feature schema changes, no new training run.
- No `train_monthly_dag` — the DAG wrapper for the PBI 20/21 training gate is dropped from the build queue.
- No drift detection, alerting, or `monitor_hourly_dag` — PBIs 23/24/25d are dropped.
- No deployment to Kubernetes, Cloud Run, or any cloud environment.
- No driver-agent simulation harness — that is PBI 28, a separate optional follow-up (`make demo-harness` should be reserved for it).
- No fix to the hardcoded `"simulated": True` in the `/heatmap` response body (line 234 of `app/api/main.py`) unless the fix is trivially co-located with the `/health` refactor. If not trivial, it is a separate clean-up.
- No `make env-init` convenience target — the manual `cp .env.example .env` step is acceptable for a portfolio demo.
- Concurrent thread-safety for `SimulatedClock` itself (`app/core/clock.py`) — that remains single-threaded and out of scope per PBI 25a.

---

## Dependencies

- **PBI 25a** (shipped, `ff466e9`) — `Clock` protocol, `WallClock`, `SimulatedClock`, and the `orchestration/` skeleton.
- **PBI 19** (shipped) — Pandera schema validation used by the `validate` task.
- **PBI 22** (shipped) — `log_predictions` used by the `log_predict` task.
- **PBI 30** (shipped) — CI workflow (ruff + mypy + pytest) that gates the PR merge.

---

## Notes for the Builder

- The technical companion to this story is `implementation_checklist/pbi25b_refresh_15min_dag_brief.md`. Section 9 of the brief gives a 12-step implementation order; follow it.

- **5 callsites, not 4.** The brief (written before a codebase audit) lists 4 clock callsites. The codebase researcher found 5: `app/api/main.py:104`, `app/api/main.py:275`, `app/ingestion/schema.py:96`, `app/monitoring/prediction_log.py:33`, and one in the ingestion/feature path. Re-audit during step 2 of the build sequence before writing the clock-discipline guard test.

- **`prediction_log._now()` already exists** as a partial wrapper. Extend it to accept an optional `Clock` parameter rather than creating a parallel helper.

- **`write_validation_log()` in `app/ingestion/schema.py`** does not accept a clock today. The refactor must add a `clock: Clock | None = None` parameter.

- **`_SpeedupClock` and `make_clock()` live in `orchestration/dags/_clock_factory.py`**, not in `app/core/`. This keeps the dependency-direction invariant (AC #12) intact: speedup is an orchestration concern, not a domain concern.

- **Scheduler heartbeat open question.** The brief (section 7) notes that Airflow's default 5-second heartbeat may not reliably trigger a `*/15 * * * *` cron at 450× speedup. The brief proposes lowering `AIRFLOW__SCHEDULER__SCHEDULER_HEARTBEAT_SEC=1` in `orchestration/.env.example`. If that is still insufficient, the accepted fallback is triggering DAG runs via an external polling loop in `make demo` rather than relying on the cron schedule under `DEMO_MODE=1`. Resolve this question before starting step 6 (DAG skeleton) and document the chosen approach in `orchestration/README.md` (AC #14).

- **Test file update reach.** Injecting `Clock` into `Predictor`, `Ingest`, `FeatureEngineer`, and `prediction_log` will require updating test fixtures in: `tests/test_inference.py`, `tests/test_feature.py`, `tests/test_serving.py`, `tests/test_monitoring.py`, and `tests/test_pbi22_acceptance.py`. All existing tests should remain green after refactor (default `clock=WallClock()` in each constructor keeps behaviour unchanged).

- **`data/watermarks/` in `.gitignore`.** Add it. Add a `.gitkeep` under `data/watermarks/refresh_15min/` so the directory tree is tracked.

- **PR target.** Base the PR against `dev_MVP`, not `main`. Use `git add -f implementation_checklist/` to force-add tracker files if needed.
