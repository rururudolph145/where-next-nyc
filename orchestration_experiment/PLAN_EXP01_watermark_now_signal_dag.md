# Implementation Plan — PBI EXP-01 (Watermark "Now-Signal" DAG)

> **Source PBI:** `orchestration_experiment/PBI_EXP01_watermark_now_signal_dag.md`
> **Stack:** Airflow **2.10.3** LocalExecutor (`docker-compose-LocalExecutor.yml`).
>   NOTE: production `orchestration/` runs Airflow 3.2.2 — different version.
> **Created:** 2026-07-14

---

## Overview

Implement a self-contained, 3-task Airflow DAG
(`read_watermark → advance_watermark → write_watermark`) that advances a JSON
watermark control file by `slice_duration` (15 min) on each run. No data
processing, no `app/` imports — the experiment stack mounts only `./dags` and
`./logs`, so the DAG MUST be fully self-contained. This is a hard constraint,
not a preference, and it resolves the PBI's OQ1 in favour of **inline**.

---

## Key findings from investigation

1. **Container isolation** — `docker-compose-LocalExecutor.yml` mounts only
   `./dags` and `./logs`; its header explicitly declares mounting `../app` and
   setting `PYTHONPATH` OUT of scope. Therefore EXP-01 cannot
   `import app.core.watermark` — inline JSON read/write is required.
2. **Format to mirror** — production `app.core.watermark` stores
   `{"position": "<naive-iso>"}` written atomically (`.tmp` + `os.replace`).
   EXP-01 writes the SAME format so a later refactor to the shared helper is a
   drop-in with zero data migration.
3. **tz discipline** — watermark stays tz-NAIVE (matches raw
   `tpep_pickup_datetime`). The DAG `start_date` is tz-aware (Airflow
   requirement) but is unrelated to the watermark; keep them separate.

---

## Affected Files

- `orchestration_experiment/dags/exp01_watermark_now_signal_dag.py` — **new**.
  The 3-task DAG with inline JSON watermark read/advance/write.
- `orchestration_experiment/data/watermark.json` — **new**. Seeded control
  file `{"position": "2024-06-03T00:00:00"}`. Optional (DAG falls back to
  `SEED_START` when absent) but created so the first run is observable.
- `orchestration_experiment/docker-compose-LocalExecutor.yml` — **update**.
  Add `- ./data:/opt/airflow/data` to `x-airflow-common.volumes` so the
  scheduler container can read/write the watermark file.
- `orchestration_experiment/PBI_EXP01_watermark_now_signal_dag.md` — **update**.
  Fill in §5 note (chosen `seed_start`/`slice_duration`, trigger instructions,
  N-run observation) and answer OQ1 (inline — required) / OQ2 (keep split).

---

## Implementation Steps

1. [ ] **Data-path decision** — add `- ./data:/opt/airflow/data` to the
   `x-airflow-common` volumes; DAG targets `Path("/opt/airflow/data/watermark.json")`.
   (Rejected alternative: writing under `/opt/airflow/dags/_state/` — pollutes
   the DAGs folder and risks the scheduler re-parsing it.)
2. [ ] **Write the DAG file** `exp01_watermark_now_signal_dag.py`:
   - `from __future__ import annotations`; stdlib only
     (`json`, `os`, `logging`, `datetime`, `timedelta`, `pathlib`) + Airflow
     (`DAG`, `PythonOperator`). No `app` / no `pandas`.
   - Constants: `WATERMARK_FILE = Path("/opt/airflow/data/watermark.json")`,
     `SEED_START = datetime(2024, 6, 3, 0, 0, 0)` (naive),
     `SLICE = timedelta(minutes=15)`.
   - `_read_watermark(**context)` — read JSON `{"position": ...}`; if file
     missing → use `SEED_START`; strip tzinfo defensively; push `window_start`
     (=W) and `window_end` (=W+SLICE) as naive ISO strings via XCom.
   - `_advance_watermark(**context)` — pull `window_end` from `read_watermark`
     XCom; return it. Kept as a SEPARATE task so EXP-02+ can insert `ingest`
     between read and advance without restructuring (OQ2 answer: keep split).
   - `_write_watermark(**context)` — pull `window_end`; atomic write
     (`.tmp` + `os.replace`) of `{"position": window_end}`. Matches production
     format exactly.
3. [ ] **DAG definition block** — `dag_id="exp01_watermark_now_signal_dag"`,
   `schedule="*/15 * * * *"`, `catchup=False`, `max_active_runs=1`,
   `start_date=datetime(2024,6,3, tzinfo=timezone.utc)`,
   `tags=["exp","watermark"]`. Wire `read >> advance >> write`;
   `trigger_rule="all_success"` on `advance` and `write` (crash-safety intent).
4. [ ] **Seed the control file** `orchestration_experiment/data/watermark.json`
   with `{"position": "2024-06-03T00:00:00"}` (and `mkdir data/`).
5. [ ] **Boot + verify** — `docker compose -f
   orchestration_experiment/docker-compose-LocalExecutor.yml up -d`; confirm
   DAG appears with NO import errors
   (`... exec airflow-scheduler airflow dags list-import-errors` FIRST — per
   the AIRFLOW QUIRK); unpause; trigger once → assert `watermark.json` ==
   `2024-06-03T00:15:00`; trigger again → `00:30:00`.
6. [ ] **Crash-safety check** — inject a failure in `_advance_watermark` (or
   before `os.replace` in `_write_watermark`) → confirm the file is unchanged.
   Remove the injected failure.
7. [ ] **Update the PBI** §5 note + OQ1 (inline — required by the mount scope)
   + OQ2 (keep advance/write split).

---

## Risks & Considerations

- **Container can't see the watermark file (highest risk)** — compose mounts
  only `./dags` + `./logs`. Step 1 must add a `./data` mount, or the DAG writes
  to an ephemeral in-container path that vanishes on restart and is invisible
  from the host. This is the one infra change; keep it to a single volume line.
- **Airflow version mismatch** — experiment is 2.10.3; production is 3.2.2.
  `PythonOperator` + `with DAG(...)` are valid in both, so EXP-01 is portable.
  Flag it: later EXP PBIs importing `app/` may hit 2.x vs 3.x API differences.
- **UID/permissions** — `logs/` is owned by UID 50000 (Airflow container user).
  The new `data/` dir must be writable by UID 50000 or `_write_watermark`
  raises `PermissionError`. Seed from host then `chown`/`chmod`, or create
  `data/` inside the container.
- **tz-naive contract** — watermark stays naive (matches raw
  `tpep_pickup_datetime`). `start_date` is tz-aware (Airflow requirement) but
  unrelated; keep separate. Inline read strips tzinfo defensively.
- **Format lock-in** — writing `{"position": "<iso>"}` atomically lets EXP-02+
  swap the inline writer for `app.core.watermark.write_watermark` with zero
  migration. Deviating from the key name now creates needless migration later.
- **No backward-compat / breaking-change risk** — EXP-01 touches nothing in
  `app/`, `orchestration/`, or shipped PBIs. Blast radius confined to
  `orchestration_experiment/`.

---

## Decisions locked (answers to PBI open questions)

- **OQ1 (reuse `app.core.watermark` vs inline)** → **inline**. Not a
  preference — the container cannot import `app` without a compose change the
  stack declares out of scope. Format kept identical for a future refactor.
- **OQ2 (merge advance + write into one task?)** → **keep separate**. The split
  is the clean insertion point for `ingest`/`validate`/`feature`/etc. in
  EXP-02+.
