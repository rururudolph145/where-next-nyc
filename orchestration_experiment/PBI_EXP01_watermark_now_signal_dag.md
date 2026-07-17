# PBI EXP-01 — Watermark "Now-Signal" DAG (Experimentation)

> **Type:** Experimentation PBI (not for release; lives under `orchestration_experiment/`)
> **Branch scope:** Local experimentation only. No changes to `orchestration/dags/` or `app/`.
> **Parent:** Orchestrator redesign brainstorm (see `MEMORY.md → orchestrator-redesign-brainstorm`).

---

## 1. Why this PBI exists

We are rebuilding the refresh DAG from scratch, one thin slice at a time, inside
`orchestration_experiment/`. Before touching any bronze/silver/gold/predict
task, we need to prove out the **simulation "now-signal"** in isolation.

Recall of the mental model (confirmed with the user 2026-07-14):

1. **"Now" signal** — a watermark `W` on disk (naive NYC timestamp) IS the
   simulator's current time. The wall clock only drives cadence.
2. **Ingestion cadence** — every DAG fire, `W` advances by `slice_duration`
   (15 min). Bronze is rewritten **cumulatively** as `[seed_start, W_new)`.
3. **Downstream** — silver → gold (features) → predict → log_predict, all in
   one DAG run.

EXP-01 covers **only step 1**. Everything else is deferred to EXP-02+.

---

## 2. Goal

Build the smallest possible DAG that can:

- Read the current watermark `W` from a control file (or seed if missing).
- Compute the next window `[W, W + slice_duration)`.
- Advance the watermark to `W + slice_duration`.
- Write it back atomically.

If this DAG runs on a schedule and the watermark advances by exactly
`slice_duration` per run, EXP-01 is done.

---

## 3. In scope

- One Airflow DAG file under `orchestration_experiment/dags/`.
- Three PythonOperator tasks (linear): `read_watermark → advance_watermark → write_watermark`.
- A local control file at `orchestration_experiment/data/watermark.json`
  (naive-NYC ISO timestamp, plus a small "advanced_at_utc" audit field).
- `slice_duration` and `seed_start` hardcoded in the DAG file (config-loading
  comes in EXP-02).
- Runs under the LocalExecutor stack in `docker-compose-LocalExecutor.yml`.

## 4. Out of scope (deferred to later EXP-PBIs)

- Bronze / silver / gold / predict / log_predict — EXP-02, 03, 04.
- `.done` sentinel / wall-clock bucket idempotency — EXP-05.
- `check_no_new_data` boundary sensor — EXP-05.
- Reuse of `app.core.watermark` helpers — TBD; we may re-implement inline
  first to keep the experiment self-contained, then refactor later.
- Any change to `app/`, `orchestration/`, `implementation_checklist/`.

---

## 5. Deliverables

1. `orchestration_experiment/dags/exp01_watermark_now_signal_dag.py`
2. `orchestration_experiment/data/watermark.json` — seeded to a chosen
   `seed_start` (e.g. `2024-06-03T00:00:00`).
3. A short note at the bottom of this PBI recording:
   - The chosen `seed_start` and `slice_duration`.
   - How to trigger the DAG locally.
   - Observation after N runs: `W_after = seed_start + N * slice_duration`.

---

## 6. DAG shape

```
read_watermark  →  advance_watermark  →  write_watermark
```

- `read_watermark` — load `data/watermark.json`; if missing, use `seed_start`.
  Push `W` and `W + slice_duration` via XCom.
- `advance_watermark` — trivially compute `W_new = W + slice_duration`.
  (Kept as a separate task so EXP-02+ can insert bronze/silver/etc. between
  read and advance without restructuring.)
- `write_watermark` — atomic write of the new watermark JSON.

DAG-level config:
- `schedule = "*/15 * * * *"` (matches the production cadence)
- `catchup = False`
- `max_active_runs = 1`
- `start_date` = a fixed past date so the DAG is enable-able immediately.

---

## 7. Acceptance (experimental, not shipping)

- After manually triggering the DAG once from a fresh state, `watermark.json`
  contains `seed_start + slice_duration`.
- After a second trigger, it contains `seed_start + 2 * slice_duration`.
- If any task inside the DAG fails, the watermark file is unchanged (crash
  safety — this is why advance and write are the last two tasks).
- No imports from `app/`. No imports from `orchestration/`.

---

## 8. Open questions — RESOLVED

- **OQ1** — Reuse `app.core.watermark` vs inline? → **INLINE, and it is
  required, not a preference.** The experiment container mounts only `./dags`,
  `./logs`, `./data`; it does NOT mount `../app` or set `PYTHONPATH`, so
  `import app.core.watermark` would `ModuleNotFoundError` at DAG parse time.
  The inline JSON format matches production `{"position": "<iso>"}` exactly for
  a future drop-in refactor.
- **OQ2** — Merge `advance_watermark` + `write_watermark`? → **Keep separate.**
  The split is the clean insertion point for `ingest`/`validate`/etc. in
  EXP-02+ (they land between `read_watermark` and `advance_watermark`).

---

## 8a. Implementation notes (2026-07-14)

**Structure (TDD-driven).** Logic lives in an Airflow-free module
`dags/exp01_watermark.py` (`read_position` / `next_window` / `write_position`),
unit-tested in `tests/test_exp01_watermark.py` (12 tests, no Airflow needed —
Airflow is NOT installed in the host test env). The DAG file
`dags/exp01_watermark_now_signal_dag.py` is a thin wrapper; its structure is
asserted in `tests/test_exp01_dag_structure.py` (SKIPS when Airflow absent).

**Dual-context import.** The DAG imports the pure module via a try/except:
package path `orchestration_experiment.dags.exp01_watermark` (host/pytest) with
a flat `exp01_watermark` fallback (container, where `/opt/airflow/dags` is on
`sys.path`). One file, two roots.

**GOTCHA — host↔container UID mismatch (cost one debug cycle).** The bind-mounted
`./data` is host-owned (UID 1000) at `0755`; the Airflow image runs as UID
50000. Read works, WRITE does not → `write_position` fails with
`PermissionError` and the watermark silently stays at seed. Fix used:
`chmod 777 data && chmod 666 data/watermark.json`. **Silver lining:** this was
an unplanned real-world test of crash-safety §7 — the failed write left
`watermark.json` untouched at seed, exactly as designed. Package `__init__.py`
markers were added to `orchestration_experiment/` and `.../dags/` so pytest can
import the module as a package.

---

## 8b. Verification note (2026-07-14) — DONE ✅

**Config chosen:** `seed_start = 2024-06-03T00:00:00` (naive),
`slice_duration = 15 min`. Watermark file: `orchestration_experiment/data/watermark.json`.

**How to run locally**
```bash
cd orchestration_experiment
docker compose -f docker-compose-LocalExecutor.yml up -d
# isolated single-run test (bypasses the scheduler / max_active_runs=1 queue):
docker compose -f docker-compose-LocalExecutor.yml exec -T airflow-scheduler \
  airflow dags test exp01_watermark_now_signal_dag 2024-06-03
cat data/watermark.json
```
Note: `dags test` keys the run by logical date, so use a DIFFERENT date each
time to force a fresh run that reads the advanced watermark.

**Observations**
| Run | Watermark before | Watermark after | Tasks |
|-----|------------------|-----------------|-------|
| 1   | `00:00:00`       | `00:15:00`      | 3× SUCCESS |
| 2   | `00:15:00`       | `00:30:00`      | 3× SUCCESS |

`W_after = seed_start + N * slice_duration` holds. Crash-safety confirmed twice:
(1) unit test `test_failed_write_leaves_prior_watermark_intact`, and (2) the
real UID-mismatch write failure left the watermark at seed until fixed.

**Test coverage:** `tests/test_exp01_watermark.py` (13 pure-module tests) +
`tests/test_exp01_dag_structure.py` (5 structural, skip-if-no-airflow; verified
green in-container). Full project suite: 452 passed / 9 skipped, no regressions.
Quality gates: ruff check + format clean, mypy clean.

---

## 9. Follow-on PBIs (preview only, not committed)

- **EXP-02** — insert `ingest` between `read_watermark` and `advance_watermark`.
  Bronze cumulative rewrite `[seed_start, W_new)`.
- **EXP-03** — add `validate` and `feature` (silver + gold).
- **EXP-04** — add `predict` and `log_predict`.
- **EXP-05** — add `.done` sentinels + `check_no_new_data` boundary sensor.
