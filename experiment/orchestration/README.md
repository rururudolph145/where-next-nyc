# Experiment: 15-minute Inference Cadence (Airflow)

Simulates a streaming-style ML pipeline at a 15-minute cadence by
replaying historical data. The simulation is split across **two DAGs**:

1. **`ingestion_15min_simulation`** — owns the simulated clock; reveals
   one gold bucket per tick.
2. **`inference_15min_simulation`** — consumes whatever ingestion has
   published; predicts that bucket; refreshes the served predictions
   parquet; hits `/heatmap`; appends to a run log.

The two DAGs talk through a marker file on disk
(`state/last_gold_bucket.json`), not through XComs or
`TriggerDagRunOperator`. This mirrors how data-driven scheduling works
in production: ingestion declares "bucket X is ready", inference picks
it up on its next tick.

> **Status**: experimentation. Lives under `experiment/` and is not part
> of the application package. Retraining and monitoring DAGs are
> deliberately deferred to keep this step small.

## Topology

```
                ┌──────────────────────────────────────────┐
   real time    │ Airflow scheduler                        │
   ──tick──────►│   * * * * * (every minute)               │
                └────────┬─────────────────────────────────┘
                         │ fires
            ┌────────────▼────────────┐
            │ ingestion_15min_dag     │  tick → ingest → clean → featurize → publish_marker
            │  (owns BucketClock)     │
            └────────────┬────────────┘
                         │ writes
              state/last_gold_bucket.json   ◄─ contract ─►
                         │
                         │ reads
            ┌────────────▼────────────┐
            │ inference_15min_dag     │  wait_for_gold → predict → smoke_test_api → log_run
            │  (consumer)             │
            └────────────┬────────────┘
                         │ writes
        data/predictions/demand_predictions.parquet ─► GET /heatmap
```

## Layout

```
experiment/orchestration/
├── README.md                       # this file
├── docker-compose.yml              # Postgres + Airflow (LocalExecutor)
├── .env.example                    # copy to .env and adjust AIRFLOW_UID
├── dags/
│   ├── ingestion_15min_dag.py      # tick + reveal-next-bucket
│   └── inference_15min_dag.py      # predict + smoke-test + log
├── src/
│   ├── bucket_clock.py             # JSON-on-disk 15-min clock (ingestion DAG only)
│   ├── ingestion_step.py           # confirm bucket in gold; publish marker
│   ├── inference_step.py           # predict one bucket; refresh parquet
│   └── run_log.py                  # CSV audit log
└── state/                          # mounted into Airflow; holds clocks + markers + log
    ├── bucket_clock.json           # current simulated bucket (written by ingestion)
    ├── last_gold_bucket.json       # what ingestion last published
    ├── last_predicted_bucket.json  # what inference last consumed
    └── run_log.csv                 # one row per inference run
```

## DAG details

### `ingestion_15min_simulation`

| Task | Job |
|---|---|
| `tick` | Advance `state/bucket_clock.json` by one 15-min step. |
| `ingest_bucket` | Placeholder for bronze-layer ingestion. In real systems, this would append the latest raw trip records. |
| `clean_bucket` | Placeholder for silver-layer cleaning. |
| `featurize_bucket` | Confirms the bucket exists in the pre-built gold parquet (`hourly_demand_featured.parquet`). |
| `publish_marker` | Writes `state/last_gold_bucket.json`. Downstream DAGs read this to know what's ready. |

> **Design note.** The existing `Ingestion`, `DataCleaner`, and
> `FeatureEngineer` classes operate on the full dataset, not per-bucket.
> Running them every simulated 15 min would be both wasteful and
> dishonest. This DAG instead reveals one bucket at a time from a
> pre-built gold parquet, preserving the bronze→silver→gold visual
> structure in the Airflow UI. A `run_strict_pipeline()` helper exists
> in `ingestion_step.py` for occasional smoke-testing the real classes.

### `inference_15min_simulation`

| Task | Job |
|---|---|
| `wait_for_gold` | Reads the marker. Skips the run if no new bucket has been published since the last prediction. |
| `predict` | Predicts demand for the bucket; appends to `data/predictions/demand_predictions.parquet` (idempotent dedup on `(pulocationid, hour)`). Optionally sleeps `INFERENCE_REAL_SLEEP_SECONDS` to slow the visible cadence. |
| `smoke_test_api` | Calls `GET /heatmap?hour=<bucket>`; logs status + zone count. Failure does NOT fail the DAG. |
| `log_run` | Appends one row to `state/run_log.csv`. |

The DAG does not own the BucketClock; it only consumes whatever
ingestion publishes. This means inference can run at any cadence
(faster than ingestion, slower than ingestion — `wait_for_gold` will
skip until ingestion advances). For the demo both DAGs are on the same
`* * * * *` schedule.

## Prerequisites

- Docker + docker compose
- Trained model at `models/model.pkl` (top-level of the repo)
- Pre-built gold layer at
  `data/gold/yellow_tripdata/hourly_demand_featured.parquet`
- Optional: FastAPI app running on the host at port 8000 so smoke tests
  return 200 instead of connection errors:
  ```bash
  uv run uvicorn app.api.main:app --port 8000
  ```

## Quick start

```bash
cd experiment/orchestration
cp .env.example .env
echo "AIRFLOW_UID=$(id -u)" >> .env

# One-time DB init + admin user creation
docker compose up airflow-init

# Bring up the stack
docker compose up -d

# Open the UI: http://localhost:8080  (admin / admin)
```

Both DAGs are unpaused on creation and will start ticking at the next
minute boundary. The ingestion DAG fires first each minute; the
inference DAG picks up the new marker on its tick.

## Tuning

Edit `.env`:

| Variable | Meaning |
|---|---|
| `INFERENCE_REAL_SLEEP_SECONDS` | Extra sleep inside `predict` (default 5). Set to `0` for full-speed replay, `60` for a more relaxed visible cadence. |
| `INFERENCE_API_BASE_URL` | Where the FastAPI app is, as seen from inside the container. Default `http://host.docker.internal:8000`. |
| `INGESTION_STRICT_MODE` | Reserved; if you switch the ingestion DAG body to call `run_strict_pipeline()`, set this to `true`. |

To change the cron, edit `schedule="* * * * *"` in either DAG file.

## Resetting the simulation

```bash
# Stop the stack
docker compose down

# Reset clocks + markers + log
rm -f state/bucket_clock.json \
      state/last_gold_bucket.json \
      state/last_predicted_bucket.json \
      state/run_log.csv

# (Optional) Reset the served predictions parquet
rm -f ../../data/predictions/demand_predictions.parquet

docker compose up -d
```

To wipe Airflow metadata too:
```bash
docker compose down -v
```

## Inspecting progress

```bash
# Watch the markers
watch -n1 'echo "=== bucket_clock ==="; cat state/bucket_clock.json; \
           echo; echo "=== last_gold_bucket ==="; cat state/last_gold_bucket.json; \
           echo; echo "=== last_predicted_bucket ==="; cat state/last_predicted_bucket.json'

# Tail the run log
tail -f state/run_log.csv
```

## Why two DAGs and not one?

- **Different cadences are coming.** Retraining is daily; monitoring is
  hourly; inference is per-bucket. Decoupling now means those DAGs slot
  in without reshaping the inference path.
- **Different failure modes.** If the gold layer rebuild fails, you
  don't want today's predictions to silently fail too — `wait_for_gold`
  will skip cleanly. If inference is slow, ingestion isn't blocked.
- **Reusable contract.** `state/last_gold_bucket.json` is a tiny,
  obvious contract. A future monitoring DAG can subscribe to the same
  marker without coordinating with inference.

## What's next (not built yet)

- `retraining_daily_dag.py` — fires once per simulated day; calls a
  `training_step.py` that wraps `app/training/trainer.py`; swaps
  `models/model.pkl` atomically.
- `monitoring_hourly_dag.py` — computes MAE/WAPE on the latest
  predictions vs actuals; can `TriggerDagRunOperator("retraining_daily")`
  on drift.
- Real per-bucket ingestion that actually reads `data/raw/...`,
  appends to bronze, and re-runs feature engineering incrementally.

## Known limitations

- Both DAGs share the `* * * * *` schedule. If the ingestion DAG runs
  slightly later than the inference DAG in the same minute, inference
  may skip until next minute. Acceptable for a demo; tighten with a
  cross-DAG sensor if needed.
- `host.docker.internal` only resolves on Docker Desktop and on Linux
  via `extra_hosts: host-gateway`. Override `INFERENCE_API_BASE_URL`
  if your setup differs.
- If the bucket clock advances past the end of the gold parquet,
  `featurize_bucket` raises `ValueError`. Reset state to start over.
