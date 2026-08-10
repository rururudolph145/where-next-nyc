# Orchestration — Airflow 2.10.3 LocalExecutor

This directory contains the production Airflow stack that hosts
`15_minute_refresh_inference_dag`, the incremental watermark-only inference
pipeline that feeds FastAPI's `/heatmap` endpoint.

## Stack

- **Airflow 2.10.3** on `apache/airflow:2.10.3-python3.11` (custom image bakes
  the pipeline's deps — see `Dockerfile` + `requirements-airflow.txt`).
- **LocalExecutor** — single-machine execution, no Redis/Celery.
- **PostgreSQL 13** metadata DB.
- **4 services**: `postgres`, `airflow-init`, `airflow-webserver`, `airflow-scheduler`.

## DAG registry

| DAG ID | PBI | Schedule | Description |
|---|---|---|---|
| `15_minute_refresh_inference_dag` (née `refresh_15min_dag`) | PBI 25b | `*/15 * * * *` | Ingestion → validation → feature → inference → log |
| `train_monthly_dag` | PBI 25c | manual (`None`) | Gold read → training → registry gate → watermark advance |
| `monitor_hourly_dag` | PBI 25d | `@hourly` | Drift detection → alert → retrain trigger (planned) |

## Bootstrap (first-time setup)

```bash
# 1. Copy the env template and set AIRFLOW_UID to your host UID.
cp orchestration/.env.example orchestration/.env
sed -i "s/^AIRFLOW_UID=.*/AIRFLOW_UID=$(id -u)/" orchestration/.env

# 2. Initialize the metadata DB (idempotent).
make airflow-init

# 3. Start the stack.
make airflow-up
```

The Airflow UI is at http://localhost:8081 (default creds: `airflow` / `airflow` —
override in `orchestration/.env` before any shared or production use).

## Running the demo

Two entrypoints:

### Static demo — `make demo`

Starts FastAPI on port 8000 serving whatever is currently under
`data/predictions/`. No Airflow required.

```bash
make demo
curl localhost:8000/heatmap
```

### Live cadence — `make demo-tick`

Fires exactly one run of `15_minute_refresh_inference_dag` against a running
Airflow stack. Each tick advances the data watermark by one slice
(`ingest.slice_duration` from `config.yml`, default 15 minutes) and
re-materializes bronze/silver/gold/predictions under `data/`.

```bash
make airflow-up          # if not already running
make demo-tick           # fires one slice
make demo-tick           # advances another slice
# ...repeat, or unpause the DAG's */15 cron for wall-clock cadence.
```

Teardown: `make demo-stop`.

## Data watermark contract

The DAG's simulation position is a single JSON control file at:

```
data/watermarks/refresh_15min/.current_watermark
```

Payload: `{"position": "<naive-iso>"}`. On the first run the DAG seeds from
`config.yml`'s `ingest.seed_start`; subsequent runs read the file and
advance by `ingest.slice_duration` per successful run. The `all_success`
trigger rule on `write_watermark` ensures a failure or `check_no_new_data`
skip leaves the watermark unchanged — the next run reprocesses the same
slice.

**Restart from scratch**: delete `data/watermarks/refresh_15min/` and the
next run reseeds from `ingest.seed_start`.

**Boundary stop**: when the watermark reaches `DATASET_MAX` (hardcoded to
`2021-03-31 23:59:57` in the DAG file), `check_no_new_data` raises
`AirflowSkipException` and the run short-circuits.

## Training DAG — `train_monthly_dag`

`train_monthly_dag` is manually triggered (`schedule=None`). It reads a 30-day
windowed slice of gold, calls `Trainer.run()`, and applies the PBI 20 registry
gate. Trigger it after at least one inference run has completed:

```bash
# From inside the container, or via the Airflow UI "Trigger DAG" button:
airflow dags trigger train_monthly_dag
```

### Dual-watermark contract

The DAG manages a **second** watermark alongside the inference one:

| File | Written by | Meaning |
|---|---|---|
| `data/watermarks/refresh_15min/.current_watermark` | inference DAG | inference position (read-only by training DAG) |
| `data/watermarks/last_training/.current_watermark` | training DAG | last completed training window end |

### Bootstrap sequencing

On the very first trigger `last_training/.current_watermark` is absent; the
DAG falls back to `ingest.seed_start` from `config.yml`. The
`check_window_ready` task then verifies that the inference watermark leads by
≥ 30 sim-days before proceeding. If it does not, the task raises
`AirflowSkipException` and the entire run short-circuits — watermark unchanged,
safe to re-trigger later.

### Fault model

| Outcome | Watermark | Model |
|---|---|---|
| Promote | advances to `last_training_wm + 30d` | new version in Production |
| Reject (gate fails) | advances to `last_training_wm + 30d` | incumbent unchanged |
| Crash (`Trainer.run` raises) | **held** (no advance) | unchanged |
| Skip (`check_window_ready`) | **held** | unchanged |

Re-triggering after a promote or reject is a no-op until the inference
watermark advances another 30 sim-days.

## Container MLflow Setup

Before the **first container training run**, you must delete any host-created
`data/mlflow.db` file:

```bash
rm -f data/mlflow.db
```

A `mlflow.db` created on the host bakes the host absolute path into the
`experiments.artifact_location` column (e.g. `/home/<user>/.../mlruns/1`).
That path is unreachable inside the container (`/opt/airflow/…`), so MLflow
can track the run but cannot find the artifact — causing `mlflow.sklearn.load_model`
and `/model-info` to degrade silently.

Deleting the file lets the container create a fresh SQLite DB on startup with
the container-relative `artifact_location`. The `model.pkl` on disk is
unaffected (it is outside the DB).

**Recovery**: if the training DAG completes but `/model-info` returns
`"status": "no_production_model"`, suspect a stale host `mlflow.db`. Stop
the stack, `rm -f data/mlflow.db`, restart, and re-trigger.

## AIRFLOW_UID

The `orchestration/.env.example` template sets `AIRFLOW_UID=1000`. Check your
UID with `id -u` and update `.env` if it differs. Using the wrong UID causes
file-ownership mismatches on mounted volumes (`dags/`, `lib/`, `logs/`).

## Stopping the stack

```bash
make airflow-down
```

## Directory layout

```
orchestration/
  docker-compose.yaml       # Airflow 2.10.3 LocalExecutor stack
  Dockerfile                # apache/airflow:2.10.3-python3.11 + pipeline deps
  requirements-airflow.txt  # pinned deps (see pyproject.toml for host parity)
  mysql.cnf                 # (unused by postgres -- legacy from EXP)
  .env.example              # copy to .env
  README.md                 # this file
  dags/
    the_15_minute_refresh_inference_dag.py    # dag_id = 15_minute_refresh_inference_dag
    train_monthly_dag.py                      # dag_id = train_monthly_dag (manual trigger)
  lib/                      # non-DAG-scanned helpers (PYTHONPATH)
    ingest_helpers.py
    validate_helpers.py
    feature_helpers.py
    predict_helpers.py
    log_predict_helpers.py
    boundary_helpers.py
    training_helpers.py
  logs/                     # gitignored, container-written
```

## Promotion methodology

New DAGs are built **incrementally** under `orchestration_experiment/` (the
"staging" tree), one PBI = one task added to a linear chain. Each PBI ships
with the full quality gate (unit + structural tests, container smoke test,
green pytest on the host). Once the DAG reaches its target shape, it is
**promoted** into `orchestration/` — the DAG file moves to `dags/`, helpers
move to `lib/`, imports flatten, EXP scope-guard tests retire, and the
promotion is captured in a design doc under `docs/superpowers/specs/`.

The promoted inference DAG's series is EXP-01..05 under
`orchestration_experiment/`. See:

- `orchestration_experiment/PBI_EXP02_ingest_bronze.md`
- `orchestration_experiment/PBI_EXP03a_validate.md`
- `orchestration_experiment/PBI_EXP03b_feature_silver_gold.md`
- `orchestration_experiment/PBI_EXP04_predict_log.md`
- `orchestration_experiment/PBI_EXP05_idempotency_boundary.md`
- `orchestration_experiment/GAP_ANALYSIS_vs_production.md`
- `docs/superpowers/specs/2026-07-24-orchestration-promotion-design.md`

The `orchestration_experiment/` tree remains a permanent playground for
future incremental DAG builds (training DAG, monitoring DAG).
