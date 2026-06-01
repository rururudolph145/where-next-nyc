# PBI 25b — `refresh_15min_dag`: Technical Brief

## 1. Summary

PBI 25b is the single PBI that converts the portfolio's **Static demo → Live cadence** mode. It adds one Airflow DAG (`refresh_15min_dag`) under the `orchestration/` scaffold shipped by PBI 25a, plus the `DEMO_MODE` factory that selects between `WallClock` and `SimulatedClock`, plus a `make demo` target that brings up the full demo stack.

No ML capability changes. No new model is trained. The deliverable is operational realism: a reviewer running `DEMO_MODE=1 make demo` can `curl /heatmap` repeatedly and watch `bucket_time` advance by 15 simulated minutes every ~2 seconds of wall-clock, exercising the exact same code path a production deployment would.

This is the highest-leverage open PBI in the tracker. It is the version of the demo that wins SmartNews / Google L4 MLE interviews because it shows the system is operationally coherent under time — not just a model wrapped in a Flask call.

---

## 2. Data Model

No data model changes.

Idempotency contract (new but no schema change): each `refresh_15min_dag` run is keyed by `(date, bucket)` where `bucket` is one of 96 quarter-hour slots in a UTC day. Re-running the same `(date, bucket)` is a no-op — the DAG checks for an existing watermark file under `data/watermarks/refresh_15min/{date}/{bucket}.done` before each task and skips if present.

---

## 3. Process Flow

### 3a. `refresh_15min_dag` task graph

```
                                       ┌──────────────────┐
                                       │  decide_bucket   │  (reads clock → emits (date, bucket))
                                       └──────────────────┘
                                                 │
                                                 ▼
                                       ┌──────────────────┐
                                       │ check_watermark  │  (skip if .done exists)
                                       └──────────────────┘
                                                 │
                          ┌──────────────────────┼──────────────────────┐
                          ▼                      ▼                      ▼
                  ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
                  │   ingest     │──────▶│   validate   │──────▶│   feature    │
                  │  (1 bucket)  │       │  (Pandera)   │       │ (Gold slice) │
                  └──────────────┘       └──────────────┘       └──────────────┘
                                                                        │
                                                                        ▼
                                                                ┌──────────────┐
                                                                │   predict    │
                                                                │ (write parq) │
                                                                └──────────────┘
                                                                        │
                                                                        ▼
                                                                ┌──────────────┐
                                                                │ log_predict  │
                                                                │  (PBI 22)    │
                                                                └──────────────┘
                                                                        │
                                                                        ▼
                                                                ┌──────────────┐
                                                                │ write_done   │
                                                                │ (watermark)  │
                                                                └──────────────┘
```

### 3b. `DEMO_MODE` factory

```python
# orchestration/dags/_clock_factory.py
def make_clock() -> Clock:
    if os.environ.get("DEMO_MODE") == "1":
        # 450× speedup: 15 sim min in ~2s of wall-clock
        start = datetime(2024, 6, 3, 0, 0, tzinfo=timezone.utc)  # a known busy Monday
        return _SpeedupClock(start=start, speedup=450)
    return WallClock()
```

`_SpeedupClock` is a thin wrapper around `SimulatedClock` (PBI 25a) that adds a background-thread `tick()` driven by `speedup * wall_clock_delta`. It is **scoped to `orchestration/`** — `app/core/clock.py` stays pure. Rationale: speedup is an orchestration concern (how fast the demo runs), not a domain concern (the model has no opinion on real-vs-sim time).

### 3c. `make demo` flow

```bash
# DEMO_MODE unset → static demo
make demo
  ├─ uv run python -m pipeline.run_all     # one-shot pipeline if predictions cache empty
  └─ uv run uvicorn app.api.main:app       # serve

# DEMO_MODE=1 → live cadence
DEMO_MODE=1 make demo
  ├─ ensure orchestration/.env present
  ├─ make airflow-up                       # bring up Airflow stack
  ├─ airflow dags unpause refresh_15min_dag
  ├─ uv run uvicorn app.api.main:app &
  └─ tail -f orchestration/logs/...
```

---

## 4. API Changes

### `app/api/main.py` — `/health` extension

The `/health` endpoint already exists (PBI 27). Extend its payload to surface the active clock:

```json
{
  "status": "ok",
  "model_version": "5",
  "model_stage": "Production",
  "simulated": true,
  "simulated_now": "2024-06-03T17:00:00+00:00",
  "wall_clock_now": "2026-05-31T14:23:11+00:00"
}
```

When `DEMO_MODE` is unset, `simulated` is `false`, `simulated_now` is omitted, and `wall_clock_now` equals `now()`.

### `/heatmap` — `bucket_time` field clarification

`/heatmap` already returns the latest predictions; under `DEMO_MODE=1` the latest prediction's `bucket_time` advances as the DAG runs. No schema change — only behavioural change. Add a doc note in the OpenAPI schema explaining that `bucket_time` reflects the clock the DAG ran under.

### Module-level Python API additions

```python
# orchestration/dags/refresh_15min_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator

with DAG(
    dag_id="refresh_15min_dag",
    schedule="*/15 * * * *",   # every 15 wall-clock minutes; under DEMO_MODE the clock runs 450× faster
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1},
) as dag:
    ...
```

```python
# orchestration/dags/_clock_factory.py
def make_clock() -> Clock: ...
class _SpeedupClock: ...  # implements Clock protocol
```

```python
# app/inference/predictor.py — minor edit
class Predictor:
    def __init__(self, ..., clock: Clock | None = None) -> None:
        self._clock = clock or WallClock()
```

(Same single-line edit applied to: `app/ingestion/ingest.py`, `app/ingestion/feature.py`, `app/monitoring/prediction_log.py`. The 4 callsites identified in PBI 25a's brief.)

### Dependency direction constraint

Unchanged from PBI 25a: `orchestration/` may import from `app/`; `app/` must never import from `orchestration/`. The `_SpeedupClock` lives in `orchestration/` precisely to keep this invariant.

---

## 5. Frontend Changes

None. Backend-only PBI.

---

## 6. Tests Required

### `tests/test_refresh_15min_dag.py` — new file (~12 tests)

**DAG integrity (extends the harness from PBI 25a)**

- `test_refresh_15min_dag_imports_cleanly` — DAG file imports without error; `dag_id` is `refresh_15min_dag`.
- `test_refresh_15min_dag_has_expected_task_ids` — task IDs match the expected list: `decide_bucket`, `check_watermark`, `ingest`, `validate`, `feature`, `predict`, `log_predict`, `write_done`.
- `test_refresh_15min_dag_schedule_is_15min` — `schedule == "*/15 * * * *"`.
- `test_refresh_15min_dag_catchup_false` — `catchup is False`.
- `test_refresh_15min_dag_max_active_runs_one` — prevents overlapping runs writing the same watermark.

**Idempotency contract**

- `test_check_watermark_skips_when_done_file_exists` — write a `.done` file, run the task, assert downstream tasks are short-circuited.
- `test_write_done_creates_watermark_with_correct_path` — completed run creates `data/watermarks/refresh_15min/{date}/{bucket}.done`.
- `test_rerun_same_bucket_is_noop` — full DAG run on a `(date, bucket)` that already has a `.done` file does not re-ingest, re-feature, or re-predict.

**Clock factory**

- `test_make_clock_returns_wallclock_when_demo_mode_unset` — monkeypatch `DEMO_MODE`; assert `isinstance(make_clock(), WallClock)`.
- `test_make_clock_returns_speedup_clock_when_demo_mode_one` — set `DEMO_MODE=1`; assert returned clock has `_speedup == 450`.
- `test_speedup_clock_advances_proportionally_to_wall_time` — drive `_SpeedupClock` for 2s of wall-time, assert sim time advances ~15min ± tolerance.

**Clock-discipline regression guard**

- `test_no_production_module_imports_datetime_utcnow_or_time_sleep` — repo-wide grep over `app/` (excluding `app/core/clock.py`); assert no occurrences of `datetime.utcnow`, `datetime.now(` without `tz=`, or `time.sleep`. This is the discipline contract PBI 25a deferred.

### `tests/test_health_endpoint_clock.py` — new file (~3 tests)

- `test_health_reports_simulated_false_when_demo_mode_unset` — payload has `simulated: false`, no `simulated_now`.
- `test_health_reports_simulated_true_with_both_clocks_when_demo_mode_one` — payload has `simulated: true`, both `simulated_now` and `wall_clock_now` are ISO-8601 UTC strings.
- `test_health_simulated_now_advances_between_calls` — under `DEMO_MODE=1`, two `/health` calls separated by 2s wall-clock show `simulated_now` advancing by ~15 min.

### `tests/test_make_demo.py` — new file (~2 tests)

- `test_make_demo_target_exists_in_makefile` — grep `makefile` for `^demo:` target.
- `test_make_demo_invokes_uvicorn` — `make --dry-run demo` output contains `uvicorn`.

---

## 7. Risks and Open Questions

- **Background-thread `_SpeedupClock` correctness.** The 450× advance is driven by a background thread that ticks `SimulatedClock.advance()` proportionally to wall-clock. `SimulatedClock` is documented as not thread-safe. The wrapper must add a `threading.Lock` around `advance()` and `now()`. Test `test_speedup_clock_advances_proportionally_to_wall_time` must use `time.sleep` (in the test, not in production) and assert with a tolerance of ±10% to absorb scheduler jitter.

- **Airflow scheduler vs `_SpeedupClock` interaction.** Airflow's scheduler ticks on wall-clock (every minute by default). The DAG schedule `*/15 * * * *` means "fire every 15 wall-clock minutes" — which under `DEMO_MODE=1` becomes "fire every 15 sim-min worth of wall-clock = ~2s." Airflow's scheduler heartbeat default is 5s; we must lower `AIRFLOW__SCHEDULER__SCHEDULER_HEARTBEAT_SEC` to 1s in `orchestration/.env` for `DEMO_MODE=1`. Document this in `orchestration/README.md`.

- **`make demo` orchestration order.** Bringing up Airflow + FastAPI + scheduler in a single `make` target requires careful sequencing: Airflow init must complete before scheduler starts; scheduler must be alive before unpausing the DAG; FastAPI can come up in parallel with the scheduler. A simple `sleep`-based sequencing is fragile; prefer `airflow db check` polling loops with a timeout. Document the failure mode (`make demo: airflow-init timed out`) in `orchestration/README.md`.

- **Predictor refactor reach.** Inserting `clock: Clock` into `Predictor.__init__` (and the 3 other callsites) touches code that was *not* refactored by PBI 25a (deliberately). Every callsite refactor is a behaviour change with a default-arg fallback (`clock or WallClock()`), so existing tests remain green — but the diff is broader than the brief's "one DAG file" framing suggests. Estimate: ~30 lines of edits across 4 files, plus ~10 test updates that pass `clock=SimulatedClock(...)` explicitly.

- **Watermark directory permissions.** `data/watermarks/refresh_15min/` does not exist yet. The DAG must create it on first run with `Path.mkdir(parents=True, exist_ok=True)`. Add `data/watermarks/` to `.gitignore`.

- **Backfill semantics.** PBI 25a's tracker entry promised `airflow dags backfill` support. The watermark contract above gives that for free (rerunning over a `(date, bucket)` range with `.done` files present is a no-op; without them, each bucket is reprocessed). Document this in `orchestration/README.md`.

- **`make demo` collision with PBI 28.** PBI 28 (Outcome Harness, deferred) will want its own `make` target. Reserve `make demo` for the live-cadence demo; PBI 28 should add `make demo-harness` or similar to avoid collision.

---

## 8. Files That Will Change

### New files

| File | Purpose |
|---|---|
| `orchestration/dags/refresh_15min_dag.py` | The DAG itself: 8 tasks, 15-min schedule, watermark idempotency |
| `orchestration/dags/_clock_factory.py` | `make_clock()` factory + `_SpeedupClock` wrapper |
| `tests/test_refresh_15min_dag.py` | DAG integrity, idempotency, clock factory tests |
| `tests/test_health_endpoint_clock.py` | `/health` payload tests under both clock modes |
| `tests/test_make_demo.py` | Makefile target presence + dry-run shape |
| `tests/test_clock_discipline.py` | Repo-wide grep guard (deferred by PBI 25a, lands here) |
| `data/watermarks/refresh_15min/.gitkeep` | Watermark directory placeholder |
| `implementation_checklist/pbi25b_refresh_15min_dag_story.md` | Approved user story (write before this brief) |
| `implementation_checklist/pbi25b_refresh_15min_dag_brief.md` | This document |

### Edited files

| File | Edit |
|---|---|
| `app/api/main.py` | Extend `/health` payload with `simulated`, `simulated_now`, `wall_clock_now` |
| `app/inference/predictor.py` | Add `clock: Clock \| None = None` to `__init__`; replace `datetime.now()` calls |
| `app/ingestion/ingest.py` | Same pattern — accept injected clock |
| `app/ingestion/feature.py` | Same pattern |
| `app/monitoring/prediction_log.py` | Same pattern |
| `makefile` | Add `demo` target (static + `DEMO_MODE=1` branches) |
| `orchestration/.env.example` | Add `AIRFLOW__SCHEDULER__SCHEDULER_HEARTBEAT_SEC=1` with comment |
| `orchestration/README.md` | Document `refresh_15min_dag`, `make demo`, backfill semantics, heartbeat tuning |
| `.gitignore` | Add `data/watermarks/` |
| `implementation_checklist/pbi_tracker.md` | Mark PBI 25b as shipped once merged |

### Deleted files

None.

---

## 9. Implementation Order

Test-first wherever possible. Each step ends GREEN before the next begins.

1. **Watermark contract** (test-first): write `test_write_done_creates_watermark_with_correct_path` (RED) → implement the watermark helper as a tiny `app/core/watermark.py` module → GREEN.
2. **Clock-discipline regression guard**: write `test_no_production_module_imports_datetime_utcnow_or_time_sleep` (likely RED today). Fix the 4 callsites to accept injected `Clock` → GREEN.
3. **`_SpeedupClock`**: write `test_speedup_clock_advances_proportionally_to_wall_time` (RED) → implement under `orchestration/dags/_clock_factory.py` with thread-lock guards → GREEN.
4. **`make_clock()` factory**: write `test_make_clock_returns_wallclock_when_demo_mode_unset` + `test_make_clock_returns_speedup_clock_when_demo_mode_one` (RED) → implement → GREEN.
5. **`/health` extension**: write the three `test_health_endpoint_clock.py` tests (RED) → extend payload → GREEN.
6. **DAG skeleton**: write the 5 DAG-integrity tests (RED) → author `refresh_15min_dag.py` with the 8 tasks → GREEN. Tasks call existing pipeline functions via thin wrappers.
7. **Idempotency contract**: write the 3 idempotency tests (RED) → wire `check_watermark` short-circuit + `write_done` final task → GREEN.
8. **`make demo` target**: write the 2 makefile tests (RED) → add `demo:` target with `DEMO_MODE` branching → GREEN.
9. **`.gitignore` + `.env.example` + `orchestration/README.md` updates** — documentation and config.
10. **Local smoke test** — `DEMO_MODE=1 make demo` end-to-end; `curl /heatmap` repeatedly; verify `bucket_time` advances; verify `/health` reports both clocks.
11. **Full quality gate** — `uv run ruff check . && uv run ruff format --check . && uv run mypy app tests && uv run pytest`.
12. **Commit + push + PR** — base `dev_MVP` (never `main`); `git add -f implementation_checklist/` for tracker files.

Suggested commit breakdown:

```
✨ feat(core): add Watermark helper for DAG idempotency
♻️ refactor(app): inject Clock into Predictor, Ingest, Feature, PredictionLog
✨ feat(orchestration): add _SpeedupClock and make_clock factory
✨ feat(api): extend /health with simulated + simulated_now + wall_clock_now
✨ feat(orchestration): add refresh_15min_dag with watermark idempotency
💚 ci(make): add demo target supporting DEMO_MODE=1
📝 docs(orchestration): document refresh_15min_dag, make demo, backfill
📝 docs(tracker): mark PBI 25b shipped
```

---

## 10. Acceptance Criteria Recap

| # | Criterion | Verified by |
|---|---|---|
| 1 | `DEMO_MODE=1 make demo` brings up FastAPI + Airflow + scheduler without manual intervention | local smoke + `test_make_demo_invokes_uvicorn` |
| 2 | `curl /heatmap` under `DEMO_MODE=1` shows `bucket_time` advancing 15 sim-min per ~2s wall-clock | local smoke |
| 3 | `/health` reports `simulated: true` with both clocks when `DEMO_MODE=1` | `tests/test_health_endpoint_clock.py` |
| 4 | `refresh_15min_dag` exists with correct `dag_id`, schedule, task IDs | `tests/test_refresh_15min_dag.py` |
| 5 | Rerunning the same `(date, bucket)` is a no-op | idempotency tests |
| 6 | No production module imports `datetime.utcnow` or `time.sleep` directly | `tests/test_clock_discipline.py` |
| 7 | `app/` does not import from `orchestration/` (dependency direction invariant) | extended from PBI 25a's existing test |
| 8 | All quality gates green: `ruff`, `mypy`, `pytest` | CI (PBI 30) |
| 9 | `orchestration/README.md` documents `make demo`, backfill, heartbeat tuning | manual review |
| 10 | Tracker updated to mark PBI 25b shipped | manual review |

---

## 11. Estimated Effort

| Step | Effort |
|---|---|
| Watermark contract + tests | 1h |
| Clock-discipline guard + 4-callsite refactor | 1.5h |
| `_SpeedupClock` + factory + tests | 1.5h |
| `/health` extension + tests | 0.5h |
| DAG skeleton + integrity tests | 2h |
| Idempotency wiring + tests | 1.5h |
| `make demo` target + tests | 1h |
| Docs + `.env.example` + `.gitignore` | 0.5h |
| Local smoke test | 1h |
| **Total** | **~10.5h (1.5 focused days)** |

---

## 12. What This PBI Does NOT Do

To keep scope honest:

- ❌ Does not train any new model. Reuses the model already in the registry.
- ❌ Does not add new features or modify the feature schema.
- ❌ Does not add drift detection, alerts, or any monitoring DAG. Those are dropped from the build queue (see tracker).
- ❌ Does not add `train_monthly_dag`. The PBI 20 gate + PBI 21 fallback already function as library calls; the DAG wrapper is dropped.
- ❌ Does not deploy to Kubernetes or any cloud. The docker-compose stack from PBI 25a is the demo environment.
- ❌ Does not add personalised ranking endpoints. `/recommend` is roadmap.
- ❌ Does not add a driver-agent simulation harness. PBI 28 is a separate, optional follow-up.

After PBI 25b ships, the portfolio is at **Demo Mode 2 — Live Cadence**, which is the recommended stopping point for a SmartNews / Google L4 MLE / SWE-ML portfolio submission.
