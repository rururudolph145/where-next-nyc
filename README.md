# New York Taxi Demand Forecasting

## The Problem

NYC yellow-taxi drivers spend roughly **a third of their shift empty** — circling, idling, or returning from outer-borough drop-offs to areas they *guess* will be busy. Guessing well is a learned skill that takes years; guessing poorly compounds into lower revenue, longer shifts, and worse service for riders who can't find a cab.

Experienced drivers already know Midtown is busy at 5 pm. The hard question is the one they *can't* answer from intuition: **where will demand be 30 minutes from now**, when I finish this drop-off — accounting for weather shifts, event let-outs, weekday-vs-holiday rhythms, and the slow drift of neighbourhoods as the city changes?

## What This System Does

This repository is an **end-to-end demand forecasting system** for NYC yellow taxis. It predicts pickup volume per taxi zone for the next 30 minutes, refreshed every 15 minutes, and serves the result as a ranked heatmap a driver app can render in one API call.

- **Audience.** Drivers / fleet dispatchers who decide *where to position next*, not *what to do right now*.
- **Decision horizon.** 30 minutes — roughly the time it takes to finish a trip and reposition.
- **Resolution.** Per taxi zone (263 zones), per 15-minute bucket — matches the granularity of the decision.
- **Counterfactual claim.** The model is meant to beat what an experienced driver already knows (the obvious time-of-day pattern). Lift is measured against a *last-week-same-hour* baseline, not against zero.

## How Value Is Measured

Model-quality metrics (MAE / RMSE / WAPE) measure the model. They don't measure whether a *driver who follows the heatmap* is better off. Two bridging metrics tie the ML output back to the business question:

| Metric | What it asks |
|---|---|
| **Recall @ top-K zones** | Of the trips that actually happened in the next 30 min, what fraction occurred in the top-K predicted zones? |
| **WAPE lift over baseline** | How much better is LightGBM than "last week, same hour, same zone"? |

A model that beats the naive baseline by a meaningful WAPE margin *and* keeps recall@3 above the baseline is doing real work. A model that only wins on MAE but offers no recall@K lift is a vanity win.

## Honest Limitations

- **Data publication lag.** NYC TLC publishes trip records on a 1–2 month lag. This repo treats the data as if it streamed in real time; the 15-minute refresh cadence is *simulated*, not live. A production deployment would need an upstream telemetry feed.
- **Single-agent assumption.** If every driver follows the same heatmap, supply outruns demand in the predicted zones and the prediction self-defeats. The system implicitly assumes the user is one of many agents, not the marginal one.
- **Macro patterns only.** The model captures temporal + spatial regularities. It does not yet incorporate weather, events, or transit disruptions — the cases where intuition most often fails. Those are roadmap items.

Built with **LightGBM**, tracked with **MLflow**, served via **FastAPI**, orchestrated with **Airflow**, and organised as a Bronze / Silver / Gold data pipeline.

## Current Forecasting Design (PBI B2)

| Knob | Value | Meaning |
|---|---|---|
| Bucket width | `15min` | Demand is aggregated into 15-minute bins per zone |
| Refresh cadence | every 15 min | A new prediction is issued at the start of each bucket |
| Target | `target_sum_t1_t2` | Sum of the next two 15-min bins = **demand over the next 30 minutes** |
| Lag features | 15min, 30min, 1h, 2h, 3h, 6h, 1d, 1w | Bucket-aligned lags of `trip_count` per zone |
| Spatial smoothing | k=5 neighbour-mean of `lag_1` | Borrows recent demand signal from the 5 geographically nearest zones (BallTree haversine) |

At time `t`, the model is shown the bucket counts up to `t` (the most recent one being `[t-15, t)`) and predicts the **cumulative pickups over `[t, t+30)`**. Strictly this is still a *nowcast* — the target window starts at `t`, so the first 15 minutes is observable-in-progress — but the `[t+15, t+30)` half is genuinely future-unknown at prediction time, so the signal is useful for driver lookahead.

A true *shifted-horizon* forecast (`[t+30, t+45)` only, no overlap with `t`) is one config flip away via the already-shipped `feature.horizon_buckets` knob (PBI A3). See `implementation_checklist/pbi_tracker.md` for the full PBI history.

## Data Pipeline (medallion architecture)

```
raw/  ──► bronze/  ──► silver/  ──► gold/  ──► model + predictions
 │           │            │           │
 │           │            │           └─ feature engineering: bucket aggregation,
 │           │            │              cyclical time, lag features, neighbour
 │           │            │              smoothing, summed-horizon target
 │           │            └─ cleaning: drop nulls, anomalous pickups, negative-duration trips
 │           └─ schema-enforced consolidation of all monthly files
 └─ daily parquets from TLC (one per day, organised by YYYY_MM/)
```

### Cleaning scope

- Drop trips with pickup outside NYC bounds
- Drop trips with negative duration (dropoff before pickup — corrupt records)
- Drop rows with nulls in the columns listed in `clean.null_subset`

Anomalous trip-attribute outliers (speed, distance, fare) are **out of scope** for this model — they don't materially affect zone-level demand prediction.

## Repository Layout

```
app/                       Production source
├── ingestion/             ingest.py · clean.py · feature.py
├── training/              trainer.py (LightGBM + MLflow logging)
├── inference/             predictor.py · simulator.py
├── api/                   FastAPI app — GET /heatmap
└── core/                  config loading, paths, logging

pipeline/                  CLI entry points (`uv run python -m pipeline.run_all`)
├── 01_ingest.py
├── 02_clean.py
├── 03_feature.py
├── 04_train.py
├── 05_inference.py
├── 06_simulate.py
└── run_all.py

tests/                     pytest suite (~85 tests, including E2E on synthetic data)
docs/                      System design + smoke-test reports (HTML)
implementation_checklist/  PBI tracker — single source of truth for design decisions
config.yml                 All runtime configuration (paths, lags, model params, …)
```

## Setup

This project uses **[uv](https://github.com/astral-sh/uv)** for environment and dependency management. Python 3.13 required.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync the venv from pyproject.toml + uv.lock
uv sync
```

## Run

```bash
# Full pipeline: ingest → clean → feature → train
uv run python -m pipeline.run_all

# Or one stage at a time
uv run python pipeline/01_ingest.py
uv run python pipeline/02_clean.py
uv run python pipeline/03_feature.py
uv run python pipeline/04_train.py
uv run python pipeline/05_inference.py
uv run python pipeline/06_simulate.py     # hourly simulation replay
```

## Serve

```bash
# Start the FastAPI heatmap service
uv run uvicorn app.api.main:app --reload
# Then: GET http://localhost:8000/heatmap
```

The `/heatmap` endpoint returns the latest per-zone predictions with `pulocationid`, `zone_name`, and `predicted_trip_count`. Sorted descending by demand.

## MLflow UI

```bash
uv run mlflow ui
# Then: http://localhost:5000
```

Training runs log model params, training/val/test metrics (MAE, RMSE, MAPE, WAPE), and the model artifact under experiment `ny_taxi_demand_model`.

## Test

```bash
uv run pytest                              # full suite
uv run pytest tests/test_feature.py -v     # one module
uv run pytest tests/test_e2e_synthetic.py  # end-to-end smoke on synthetic data

# Lint + type-check
uv run ruff check app/ tests/
uv run mypy app/ tests/
```

## Configuration

All runtime knobs live in `config.yml`. Highlights:

| Section | Key | Purpose |
|---|---|---|
| `ingest` | `start_year_month` / `end_year_month` | Date range to pull from raw |
| `feature` | `bucket` | Aggregation bin width (`"15min"`, `"30min"`, `"h"`) |
| `feature` | `lags` | Lag offsets in *buckets* |
| `feature` | `horizon_sum_buckets` | N future buckets to sum into the target (PBI B2) |
| `feature` | `horizon_buckets` | Single-bucket shifted-horizon target (PBI A3) |
| `feature` | `n_neighbors` | k for spatial neighbour-mean smoothing (PBI B1) |
| `model` | `params` | LightGBM hyperparameters |
| `train` | `features` / `target` | Feature list + target column name |

The legacy keys `feature.lag_hours` and `bucket: "h"` remain accepted for back-compat with older configs and notebooks.

## Docker

```bash
docker build -t ny-taxi-demand .
docker run -p 80:80 ny-taxi-demand
```

## Documentation

- `docs/system_design.html` — overall architecture
- `docs/design_t30_15min_refresh.html` — t+30 / 15-min refresh design (B-track rationale)
- `docs/phase6_smoke_test_report.html` — end-to-end smoke report
- `implementation_checklist/pbi_tracker.md` — PBI history, dependencies, priority order

## Roadmap (high level)

- **MVP (shipped)** — single-horizon training, inference, heatmap API
- **B-track (shipped)** — 15-min bucket + spatial smoothing + 30-min summed-horizon target
- **Performance hardening (in progress)** — parallel Bronze ingestion, `/heatmap` pagination, streaming Bronze write
- **Post-MVP** — personalised ranking (DiDi-style), simulated near-real-time inference, model drift monitoring
