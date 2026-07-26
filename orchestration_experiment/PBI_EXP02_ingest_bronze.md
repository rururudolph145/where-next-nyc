# PBI EXP-02 — Ingest Layer (Bronze, Cumulative) (Experimentation)

> **Type:** Experimentation PBI (not for release; lives under `orchestration_experiment/`)
> **Depends on:** EXP-01 (watermark now-signal DAG must advance correctly).
> **Parent:** Orchestrator redesign brainstorm (see `MEMORY.md → orchestrator-redesign-brainstorm`).

---

## 1. Why this PBI exists

EXP-01 proved the simulation "now-signal" (watermark) advances by exactly
`slice_duration` per run. EXP-02 inserts the **first real processing task**:
ingesting raw NYC taxi data into **bronze**.

This is where the mental model's step 2 lands:

> Every DAG fire, `W` advances by `slice_duration` (15 min). Bronze is
> rewritten **cumulatively** as `[seed_start, W_new)` — NOT an incremental
> 15-min append. Only the upper bound moves; the lower bound is frozen at
> `seed_start` (ADR-001 single-file full rewrite).

---

## 2. Goal

Insert an `ingest` task **between** `read_watermark` and `advance_watermark`
so a DAG run reads the window, writes cumulative bronze up to `window_end`,
then advances the watermark — but only if ingest succeeded.

---

## 3. In scope

- Add `ingest` task to `orchestration_experiment/dags/`, wired as:
  `read_watermark → ingest → advance_watermark → write_watermark`.
- Introduce config loading (`app.core.config.load_config`) so `seed_start`,
  `slice_duration`, and `data.bronze_path` come from config, not hardcoded.
  (This is the EXP-02 answer to EXP-01's OQ1 — we start depending on `app.core`.)
- Call `Ingestion().run(window_start=seed_start, window_end=window_end)` where
  `seed_start` is FROZEN from config and `window_end` comes from the XCom.
- Bronze written to the experiment's own data root (keep it isolated from the
  production `data/` tree; use e.g. `orchestration_experiment/data/bronze/`).

## 4. Out of scope (deferred)

- Schema validation — EXP-03.
- Silver / gold feature engineering — EXP-03.
- Predict / log — EXP-04.
- `.done` sentinels + `check_no_new_data` boundary — EXP-05.
- Any change to `app/`, `orchestration/`, `implementation_checklist/`.

---

## 5. Deliverables

1. Updated DAG file with the `ingest` task inserted.
2. Bronze parquet materialized under the experiment data root after a run.
3. Note recording: bronze row count grows monotonically across runs;
   `min(tpep_pickup_datetime) == seed_start`, `max < window_end`.

---

## 6. DAG shape

```
read_watermark → ingest → advance_watermark → write_watermark
```

- `ingest` — pull `window_end` from `read_watermark` XCom, freeze
  `window_start = seed_start` from config, run cumulative bronze rewrite.
- `advance_watermark` keeps `trigger_rule="all_success"` — a failed ingest
  leaves the watermark unchanged so the next run retries the same slice.

---

## 7. Acceptance (experimental)

- After run 1: bronze exists, covers `[seed_start, seed_start + slice)`.
- After run 2: bronze covers `[seed_start, seed_start + 2*slice)` — a strict
  superset of run 1.
- Watermark advances in lockstep (verify EXP-01 behaviour still holds).
- Forced ingest failure → watermark unchanged, bronze from prior run intact.
- No imports from `orchestration/`.

---

## 8. Open questions (log answers here as we go)

- **OQ1 — RESOLVED (isolated).** Bronze is written to the experiment's own root
  (`orchestration_experiment/data/` → `/opt/airflow/data_exp/` in-container),
  never the production `data/` tree. Raw is a **read-only** mount
  (`../data/raw:/opt/airflow/data/raw:ro`), so the 340MB source cannot be
  clobbered while iterating.
- **OQ2 — RESOLVED (inject a config dict).** `Ingestion(config=...)` is
  injectable; `build_ingest_config()` deep-overrides only `data.raw_path` /
  `data.bronze_path` to absolute experiment paths. No experiment `config.yml`
  fork — the mounted `config.yml` supplies seed/slice/schema values, the
  override supplies the isolated paths. Avoids CWD-relative path ambiguity.

---

## 9. Verification note — GREEN (2026-07-14)

**Stack:** Airflow 2.10.3 LocalExecutor via custom Dockerfile
(`FROM apache/airflow:2.10.3-python3.11`, baked `pandas/numpy/pyarrow/pyyaml`).
`app/`, `config.yml`, and `data/raw` bind-mounted read-only; bronze written to
the writable experiment root. `PYTHONPATH=/opt/airflow`.

**DAG shape (verified):** `read_watermark → ingest → advance_watermark →
write_watermark` — 4 tasks, linear, no import errors
(`airflow dags list-import-errors` → "No data found").

**Cumulative bronze (AC §7, verified via `dags test`):**

| Run | Slice read | Ingest window (frozen seed) | Bronze rows | min pickup | max pickup | Watermark after |
|-----|-----------|-----------------------------|-------------|-----------|-----------|-----------------|
| 1 (`01-18`) | `[00:00, 00:15)` | `[00:00, 00:15)` | 105 | 00:00:05 | 00:14:58 | `00:15` |
| 2 (`01-19`) | `[00:15, 00:30)` | `[00:00, 00:30)` | **183** (strict superset) | 00:00:05 | 00:29:49 | `00:30` |

`min(pickup) ≥ seed_start` and `max(pickup) < window_end` hold both runs; the
frozen lower bound keeps `min` at the seed while rows grow monotonically. (Note:
`min == seed_start` exactly is NOT expected — the first trip is at 00:00:05, not
00:00:00; the contract is `>= seed_start`.)

**Crash safety (AC §7, verified):** forced `save_bronze` to fail
(`PermissionError: [Errno 13]` on a read-only bronze dir). Ingest task FAILED;
because `advance`/`write` are `trigger_rule="all_success"`, the watermark stayed
at its prior position — the next run reprocesses the same slice.

**Gates:** host `ruff check` clean, `ruff format --check` clean, `pytest`
462 passed / 10 skipped (the +1 skip vs. EXP-01 is
`test_exp02_dag_structure.py`, correctly skipping — Airflow absent on host).
EXP-02 unit helpers at 100% coverage.

**Deviations from PLAN_EXP02 (flagged):**
1. Dropped `pandera>=0.20` from the image deps — tracing `sys.modules` showed the
   ingest path never imports it (validation is EXP-03). YAGNI.
2. Added `pyyaml>=6.0` — the plan omitted it, but
   `Ingestion → load_config → yaml.safe_load` needs it or the container import
   crashes at parse time.

**No imports from `orchestration/`** (AC §7): the DAG imports only `app.*` and
the experiment's own pure helpers.
