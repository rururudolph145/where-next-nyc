# Orchestration — Airflow 3.2.2 Platform

This directory contains the Airflow 3.2.2 infrastructure for the NYC Taxi Demand
Forecasting pipeline (C00 Orchestrator lane). PBI 25b ships `refresh_15min_dag`;
the remaining two planned DAGs are documented below.

## Planned DAGs

| DAG ID | PBI | Schedule | Description |
|---|---|---|---|
| `refresh_15min_dag` | PBI 25b | `*/15 * * * *` | Ingestion → validation → feature → inference → log |
| `train_monthly_dag` | PBI 25c | `@monthly` | Training + PBI 20 registry gate |
| `monitor_hourly_dag` | PBI 25d | `@hourly` | Drift detection → alert → retrain trigger |

## Bootstrap (first-time setup)

```bash
# 1. Copy the env template and set AIRFLOW_UID to your local user ID.
#    WARNING: override any credentials before production use.
cp orchestration/.env.example orchestration/.env

# 2. Initialize the Airflow metadata database (idempotent — safe to re-run).
make airflow-init

# 3. Start the full stack in the background.
make airflow-up
```

The Airflow UI is reachable at http://localhost:8090 after `make airflow-up`.
Default credentials from the upstream compose file are **airflow / airflow** —
override these in `orchestration/.env` before any shared or production deployment.

## Stopping the stack

```bash
make airflow-down
```

This stops all containers without destroying named volumes. Re-running
`make airflow-up` picks up where you left off.

## Source

`orchestration/docker-compose.yaml` was fetched from:
https://airflow.apache.org/docs/apache-airflow/3.2.2/docker-compose.yaml

The file is verbatim **with one deliberate divergence**: the `airflow-apiserver`
service's host port is remapped from `8080` to `8090` to avoid a common
collision with Windows-side IIS / Spring Boot / Tomcat defaults on developer
machines. Search for `PBI 25a:` in the compose file to find the single modified
line. The container's internal port (and the healthcheck) remains `8080`.

To pin a different Airflow version, re-fetch the file from the upstream URL for
the desired version and re-apply the port remap.

## Running the demo

Two modes are available via the `make demo` target in the project root.

### Static demo (default)

```bash
make demo
```

Starts FastAPI serving cached predictions via uvicorn on port 8000. No Airflow
stack is launched. `bucket_time` does not change between `/heatmap` calls.

### Live cadence demo (`DEMO_MODE=1`)

```bash
cp orchestration/.env.example orchestration/.env   # first time only
make demo DEMO_MODE=1
```

This brings up the full Airflow compose stack and then runs a host-side
**external trigger loop** that fires `refresh_15min_dag` every 2 seconds. At
the configured 450x clock speedup, each trigger covers one 15-minute simulated
bucket. Press **Ctrl-C** to stop the loop.

The external trigger approach was chosen over relying on the cron schedule
(`*/15 * * * *`) under DEMO_MODE because Airflow's scheduler heartbeat (default
5 seconds) cannot reliably fire a cron at 450x simulated speed. The cron
schedule remains correct for production (wall-clock) deployments.

To tear down the compose stack:

```bash
make demo-stop
```

## Backfill semantics

`refresh_15min_dag` uses **`catchup=False`** -- Airflow does not backfill
missed intervals on startup. Combined with the watermark contract (below), this
means:

- **First run**: processes the current `(day, bucket)` determined by the active
  clock.
- **Re-run of the same bucket**: the `check_watermark` task detects the existing
  `.done` file at `data/watermarks/refresh_15min/{date}/{bucket:02d}.done` and
  raises `AirflowSkipException`. All downstream tasks skip. The run is a no-op.
- **Manual backfill**: delete the `.done` files for the desired range, then
  trigger the DAG for each bucket. Each run processes one bucket and writes a
  new `.done` file on success.

If a run fails between `check_watermark` and `write_done`, no `.done` file is
written, so the next trigger retries that bucket automatically.

**Edge case**: if a `.done` file exists but the corresponding Gold parquet was
manually deleted, the DAG treats the bucket as complete and skips it. This is
intentional -- a missing Gold file after a confirmed run is a manual-intervention
case.

## AIRFLOW_UID

The `orchestration/.env.example` template sets `AIRFLOW_UID=1000`. This is the
default UID on most single-user Linux and WSL systems. If your host UID differs,
update the value in `orchestration/.env` after copying:

```bash
# Check your UID:
id -u

# Then edit orchestration/.env:
AIRFLOW_UID=<your-uid>
```

Using the wrong UID causes file-ownership mismatches on the volume-mounted
`dags/`, `logs/`, `plugins/`, and `config/` directories.

## Known tuning items

The upstream compose file ships with two settings appropriate for local
development but requiring override before any production deployment:

- **Airflow example DAGs enabled** (`AIRFLOW__CORE__LOAD_EXAMPLES=true` in the
  compose file). Set to `false` in `orchestration/.env` to suppress example DAGs.
- **Default Fernet key.** Set `AIRFLOW__CORE__FERNET_KEY` in `orchestration/.env`
  to a key generated by `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`.

## Directory layout

```
orchestration/
  docker-compose.yaml   Airflow 3.2.2 compose file (verbatim except for one
                        documented port remap; see "Source" section)
  .env.example          Template — copy to .env and customize
  .env                  Local environment (gitignored — never commit)
  README.md             This file
  dags/                 DAG files volume-mounted into scheduler + worker
  plugins/              Custom Airflow plugins
  config/               Airflow config overrides (airflow.cfg or env files)
  logs/                 Task logs (gitignored)
```
