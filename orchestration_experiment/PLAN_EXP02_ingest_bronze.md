# Implementation Plan — PBI EXP-02 (Ingest Layer, Bronze Cumulative)

> **Source PBI:** `orchestration_experiment/PBI_EXP02_ingest_bronze.md`
> **Depends on:** EXP-01 (watermark now-signal DAG — GREEN, verified 2026-07-14).
> **Stack:** Airflow **2.10.3** LocalExecutor, now via a **custom Dockerfile**.
> **Created:** 2026-07-14

---

## Overview

Insert an `ingest` task between `read_watermark` and `advance_watermark` so a
DAG run reads the window, writes **cumulative** bronze `[seed_start, window_end)`
via the real `app.ingestion.Ingestion`, then advances the watermark only if
ingest succeeded. This is the first EXP step that needs `app/` + heavy deps
(pandas/pyarrow/pandera) inside the container, so it also fattens the stack from
the stock image to a baked custom image.

---

## Decisions locked (with the user, 2026-07-14)

1. **Container wiring** → **custom Dockerfile** (`FROM apache/airflow:2.10.3`,
   bake app deps). Reproducible, fast restarts, portfolio-credible.
2. **Seed timeline** → **`2021-01-18T00:00:00` from `config.yml`**. The raw data
   is 2021-01..03; EXP-01's `2024-06-03` placeholder would ingest 0 rows.
   Overwrite `data/watermark.json` to the 2021 seed.
3. **Data roots** → raw **read-only** input, bronze written to the **experiment
   root**. Inject a config override into `Ingestion(config=...)` so it never
   touches the production `data/` tree.

---

## Key findings from investigation

- `Ingestion(config: dict | None = None)` — accepts an **injectable config**.
  This is the isolation seam: pass absolute experiment paths for `raw_path` /
  `bronze_path` so the container never uses CWD-relative production paths.
- `Ingestion.run(window_start, window_end)` — both NAIVE; cumulative bronze is
  `run(window_start=seed_start, window_end=window_end)` (lower bound FROZEN at
  seed, upper bound from the advancing watermark). Matches the old prod DAG.
- `Ingestion.__init__` asserts `ingest.slice_duration == feature.bucket`
  (both `15min` in config) — any override config MUST keep them equal.
- `load_config(path="config.yml")` resolves relative to **CWD** (`/opt/airflow`
  for Airflow tasks). We bypass CWD ambiguity by injecting the config dict.
- Raw layout confirmed: `data/raw/yellow_tripdata/YYYY_MM/YYYY_MM_DD.parquet`;
  `2021_01/` exists → seed `2021-01-18` finds real files. Raw is **340 MB**.
- Deps needed in-container: `pandas>=2.2, numpy>=2, pyarrow>=18, pandera>=0.20`
  (geopandas NOT needed for ingest — defer to when a task requires it).

---

## Affected Files

- `orchestration_experiment/Dockerfile` — **new**. `FROM
  apache/airflow:2.10.3-python3.11`; `pip install` the ingest deps.
- `orchestration_experiment/requirements-exp.txt` — **new**. Pinned dep list
  the Dockerfile installs (pandas/numpy/pyarrow/pandera).
- `orchestration_experiment/docker-compose-LocalExecutor.yml` — **update**.
  Swap `image:` → `build: .` on `x-airflow-common`; add mounts:
  `../app:/opt/airflow/app:ro`, `../config.yml:/opt/airflow/config.yml:ro`,
  `../data/raw:/opt/airflow/data/raw:ro`; keep `./data:/opt/airflow/data_exp`
  writable for bronze; set `PYTHONPATH: /opt/airflow`.
- `orchestration_experiment/dags/exp02_ingest.py` — **new** pure-ish helper:
  `build_ingest_config(base, *, raw_root, bronze_root)` returning the override
  dict, and `run_ingest(window_end, *, seed_start, config)` calling
  `Ingestion(config=...).run(...)`. Keeps XCom/Airflow out of the tested logic.
- `orchestration_experiment/dags/exp02_ingest_bronze_dag.py` — **new** DAG:
  `read_watermark → ingest → advance_watermark → write_watermark`. Same
  dual-context import pattern as EXP-01.
- `orchestration_experiment/data/watermark.json` — **update**. Reseed to
  `2021-01-18T00:00:00`.
- `tests/test_exp02_ingest.py` — **new**. Unit tests for the pure helpers
  (config override shape, path isolation, window bounds) with a MOCKED
  `Ingestion` (no real 340MB read in tests — testing rule: no external data).
- `tests/test_exp02_dag_structure.py` — **new**. 4-task structural test
  (skip-if-no-airflow), mirroring EXP-01.
- `orchestration_experiment/PBI_EXP02_ingest_bronze.md` — **update**. Answer
  OQ1/OQ2, add verification note.

---

## Implementation Steps

1. [x] **RED** — write `tests/test_exp02_ingest.py` first:
   - `build_ingest_config` returns a dict whose `data.raw_path` /
     `data.bronze_path` point at the injected roots (isolation), and preserves
     `ingest.slice_duration == feature.bucket`.
   - `run_ingest` calls `Ingestion(config=...).run(window_start=seed_start,
     window_end=window_end)` with the FROZEN seed as the lower bound (assert via
     a mocked `Ingestion`). Confirm tests fail (module missing).
2. [x] **GREEN** — implement `dags/exp02_ingest.py` (pure helpers, no Airflow
   import) to pass the tests.
3. [x] **Dockerfile + requirements** — write `requirements-exp.txt` (pinned) and
   `Dockerfile`; edit compose to `build: .` + the four mounts + `PYTHONPATH`.
4. [x] **DAG wrapper** — `dags/exp02_ingest_bronze_dag.py`: thin `_ingest` task
   pulls `window_end` from XCom, freezes `seed_start` from config, calls
   `run_ingest`. Wire `read → ingest → advance → write`; `advance`/`write` keep
   `trigger_rule="all_success"`. Add structural test.
5. [x] **Reseed** `data/watermark.json` → `2021-01-18T00:00:00`; ensure
   `data/bronze/` writable by UID 50000 (chmod, per EXP-01 gotcha).
6. [x] **Build + boot** — `docker compose ... build` then `up -d`; FIRST check
   `airflow dags list-import-errors` (the AIRFLOW QUIRK). Confirm the DAG parses
   with `app` importable.
7. [x] **Smoke test** — `dags test exp02_ingest_bronze_dag 2021-01-18`:
   - watermark advances `2021-01-18T00:00 → 00:15`;
   - bronze parquet exists under the experiment root, non-empty,
     `min(tpep_pickup_datetime) == seed_start`, `max < window_end`;
   - second run (`2021-01-19`) → watermark `00:30`, bronze is a **superset**
     (row count grew).
8. [x] **Crash-safety** — force `ingest` to fail → watermark unchanged, prior
   bronze intact.
9. [x] **Gates + PBI note** — ruff, mypy (pure helper + tests), full pytest;
   verify no regressions; fill the PBI verification note.

---

## Risks & Considerations

- **Container fattening / build time (new)** — first `docker build` pulls
  pandas/pyarrow (~hundreds of MB). One-time cost; cached thereafter. Pin
  versions in `requirements-exp.txt` to match `pyproject.toml` (numpy>=2,
  pandas>=2.2, pyarrow>=18, pandera>=0.20) to avoid drift from the host env.
- **CWD-relative config paths** — `load_config` and the paths inside it are
  CWD-relative. Mitigation: inject an override config with ABSOLUTE experiment
  paths into `Ingestion(config=...)`; do not rely on the mounted `config.yml`
  for path resolution (only for the seed/slice/schema values).
- **Cumulative rewrite cost (performance rule)** — bronze is a full rewrite of
  `[seed, window_end)` every run (ADR-001). At 15-min slices over 340MB raw
  this is cheap early but grows; fine for the experiment. Note it; do NOT
  optimize (YAGNI) — incremental append is explicitly out of scope.
- **`slice_duration == feature.bucket` guard** — the override config must keep
  these equal or `Ingestion.__init__` asserts. Both are `15min`; keep it.
- **Read-only raw mount** — `../data/raw:...:ro` protects the source. Bronze
  goes to the writable experiment root only. Verify the DAG never writes under
  `/opt/airflow/data/raw`.
- **UID 50000 write perms** — same gotcha as EXP-01: the writable bronze dir
  must be writable by the container user (chmod 777 the experiment data dir).
- **No test reads real data (testing rule)** — unit tests MUST mock
  `Ingestion`; the 340MB read happens only in the container smoke test, never in
  pytest. Keeps the suite fast and hermetic.
- **Airflow 2.10 vs prod 3.2.2** — unchanged from EXP-01; portable APIs only.
- **Blast radius** — confined to `orchestration_experiment/` + two new test
  files. No change to `app/`, `orchestration/`, `config.yml`, or `data/`.

---

## Open questions to resolve during build (from PBI EXP-02 §8)

- **OQ1** → isolated experiment data root: **YES** (locked above).
- **OQ2** → `Ingestion` path override: **inject config dict** (locked above);
  confirmed viable because `Ingestion(config=...)` is injectable.
