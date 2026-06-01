# New York Taxi Demand Forecasting

[![Lint](https://github.com/rururudolph145/NewYorkTaxiDemandForecasting/actions/workflows/lint.yml/badge.svg)](https://github.com/rururudolph145/NewYorkTaxiDemandForecasting/actions/workflows/lint.yml)
[![Tests](https://github.com/rururudolph145/NewYorkTaxiDemandForecasting/actions/workflows/test.yml/badge.svg)](https://github.com/rururudolph145/NewYorkTaxiDemandForecasting/actions/workflows/test.yml)

> **End-to-end demand forecasting for NYC yellow taxis.** Predicts pickup volume per taxi zone for the next 30 minutes, refreshed every 15 minutes, served as a ranked heatmap a driver app can render in one API call. Model lift is measured against a *last-week-same-hour* baseline — beating zero is meaningless.

---

## The Problem

NYC yellow-taxi drivers spend roughly **a third of their shift empty** — circling, idling, or returning from outer-borough drop-offs to areas they *guess* will be busy. Guessing well is a learned skill that takes years; guessing poorly compounds into lower revenue, longer shifts, and worse service for riders who can't find a cab.

Experienced drivers already know Midtown is busy at 5pm. The hard question is the one they *can't* answer from intuition: **where will demand be 30 minutes from now**, when I finish this drop-off — accounting for weather shifts, event let-outs, weekday-vs-holiday rhythms, and the slow drift of neighbourhoods as the city changes?

This repository answers that question for the next 30 minutes, per taxi zone, refreshed every 15 minutes.

---

## How To Run This Project (pick a demo mode)

This project is structured around **demo modes** — independently runnable demonstrations, each with its own reviewer-facing story. Pick the mode that matches how much time you have to look at the repo.

| Demo mode | Command | What you'll see | Status |
|---|---|---|---|
| **Static demo** | `make demo` | One curl returns the ranked heatmap with baseline-lift metrics, `/health`, `/model-info` | ✅ Available now |
| **Live cadence** | `DEMO_MODE=1 make demo` | Same API, but the heatmap visibly refreshes every ~2 seconds = 15 simulated minutes — Midtown at sim 5pm, JFK at sim 6am | 🟡 Next (PBI 25b) |
| **Closed training loop** | — | Manual retrain → champion-vs-challenger gate → register or roll back | 🔵 Designed (see Future Work) |
| **Closed feedback loop** | — | Drift detection → alert → auto-retrain trigger | 🔵 Designed (see Future Work) |
| **Outcome harness** | — | Synthetic drivers replay a held-out week; per-policy trips-served lift table | 🔵 Designed (see Future Work) |
| **Personalised ranking** | — | `/recommend?lat=&lon=` returns top-N zones by predicted demand / travel cost | ⚪ Roadmap |

No demo mode is "better" than another — each is a different lens on the same system. The first two modes are the demonstrable proofs; the remaining four are scoped designs documented at the bottom of this README.

---

## Demo Mode 1 — Static Demo (available now)

The driver-facing question is answered end-to-end on a static historical dataset. Bronze → Silver → Gold pipeline, LightGBM validated against a last-week-same-hour baseline, FastAPI heatmap endpoint with fallback to the last approved model.

```bash
# One-time setup
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv if needed
uv sync                                            # sync the venv from uv.lock

# Run the static demo
make demo
# → trains (or loads cached model), starts FastAPI on :8000, prints sample curl
curl http://localhost:8000/heatmap | jq '.[:5]'    # top-5 zones for the latest bucket
curl http://localhost:8000/health                  # liveness + sim-clock flag
curl http://localhost:8000/model-info              # current model version + stage
```

### What the static demo proves

- **The model adds real lift over what drivers already know.** Lift is measured against a *last-week-same-hour* baseline (PBI 18). See the table below.
- **The serving path is honest about failure.** If model loading fails, inference falls back to the previously approved registry version (PBI 21). `/model-info` always reports which version actually served the request.
- **Every code change is gated.** PRs run `pytest`, `ruff`, `mypy` via GitHub Actions (PBI 30).

### Headline numbers

| Metric | LightGBM | Last-week-same-hour baseline | Lift |
|---|---|---|---|
| WAPE | _populated by PBI 18 reporting step_ | | |
| MAE | | | |
| Recall @ top-3 zones | | | |
| Recall @ top-5 zones | | | |

> A model that beats the baseline by a meaningful WAPE margin *and* keeps recall@K above the baseline is doing real work. A model that wins only on MAE but offers no recall@K lift is a vanity win.

### Sample heatmap output

![Sample heatmap — top-5 predicted zones for a weekday 5pm bucket](docs/assets/sample_heatmap.png)

---

## Demo Mode 2 — Live Cadence (next; PBI 25b)

```bash
DEMO_MODE=1 make demo
# → starts FastAPI + Airflow scheduler with SimulatedClock(speedup=450)
curl http://localhost:8000/heatmap    # bucket_time advances 15 sim minutes per ~2s wall-clock
curl http://localhost:8000/health     # → {"simulated": true, "simulated_now": "...", "wall_clock_now": "..."}
```

### What changes versus the static demo

Adds no ML capability — adds **operational realism.** The system stops being a one-shot script and becomes a continuously-refreshing service driven by an injected `Clock` (PBI 25a). Every component that asks "what time is it?" — Airflow sensors, ingestion watermarks, the predictor's "current bucket", the API's "latest hour" lookup — reads from the injected clock, never from the OS.

| Implementation | When | Behaviour |
|---|---|---|
| `WallClock` | production (`DEMO_MODE` unset) | wraps real time |
| `SimulatedClock(start, speedup=450)` | demo (`DEMO_MODE=1`) | 15 simulated minutes per ~2 seconds wall-clock |

### Why simulate the clock at all?

NYC TLC publishes trip records on a 1–2 month lag. A "live" production deployment isn't possible from a portfolio repo — but it *is* honestly demonstrable on a compressed simulated clock that exercises the exact same code path production would. The `/health` endpoint reports both `simulated_now` and `wall_clock_now` so nothing about the simulation is hidden.

This is the version of the demo that proves the system is **operationally coherent under time** — not just a model wrapped in a Flask call.

---

## How the System Works

```
raw/  ──► bronze/  ──► silver/  ──► gold/  ──► model + predictions
 │           │            │           │
 │           │            │           └─ feature engineering: bucket aggregation,
 │           │            │              cyclical time, lag features, neighbour
 │           │            │              smoothing, summed-horizon target
 │           │            └─ cleaning: drop nulls, anomalous pickups, negative-duration trips
 │           └─ schema-enforced consolidation of all monthly files (Pandera, PBI 19)
 └─ daily parquets from TLC (one per day, organised by YYYY_MM/)
```

- **Audience.** Drivers / fleet dispatchers who decide *where to position next*, not *what to do right now*.
- **Decision horizon.** 30 minutes — roughly the time it takes to finish a trip and reposition.
- **Resolution.** Per taxi zone (263 zones), per 15-minute bucket — matches the granularity of the decision.

### Forecasting target

| Knob | Value | Meaning |
|---|---|---|
| Bucket width | `15min` | Demand aggregated into 15-minute bins per zone |
| Refresh cadence | every 15 min | A new prediction is issued at the start of each bucket |
| Target | `target_sum_t1_t2` | Sum of the next two 15-min bins = demand over the next 30 minutes |
| Lag features | 15min, 30min, 1h, 2h, 3h, 6h, 1d, 1w | Bucket-aligned lags of `trip_count` per zone |
| Spatial smoothing | k=5 neighbour-mean of `lag_1` | Borrows recent demand signal from the 5 geographically nearest zones (BallTree haversine) |

Strictly this is an *extended nowcast* — the target window starts at `t`, so the first 15 minutes overlap prediction time — but the `[t+15, t+30)` half is genuinely future-unknown at prediction time, so the signal is useful for driver lookahead. A true *shifted-horizon* forecast (`[t+30, t+45)` only) is one config flip away via the `feature.horizon_buckets` knob (PBI A3, already shipped).

### Stack

Built with **LightGBM**, tracked with **MLflow**, served via **FastAPI**, orchestrated with **Airflow** (on a `SimulatedClock` in demo mode), organised as a Bronze / Silver / Gold data pipeline.

---

## Repository Layout

```
app/                       Production source
├── ingestion/             ingest.py · clean.py · feature.py · schema.py (Pandera, PBI 19)
├── training/              trainer.py (LightGBM + MLflow + registry gate, PBI 20)
├── inference/             predictor.py (with fallback, PBI 21) · simulator.py
├── monitoring/            prediction_log.py · actuals_join.py (PBI 22)
├── api/                   FastAPI app — GET /heatmap · GET /health · GET /model-info
└── core/                  clock.py (PBI 25a) · config · paths · logging · metrics

orchestration/             Airflow scaffold (PBI 25a)
├── docker-compose.yaml    Verbatim Airflow 3.2.2 compose
├── dags/                  refresh_15min_dag (PBI 25b — open)
└── README.md              Bootstrap + named DAG registry

pipeline/                  CLI entry points (`uv run python -m pipeline.run_all`)
tests/                     pytest suite — unit · acceptance · E2E on synthetic data
docs/                      System design + readiness checklists (HTML + MD)
implementation_checklist/  PBI tracker — single source of truth for design decisions
config.yml                 All runtime configuration (paths, lags, model params, …)
```

---

## Setup

This project uses **[uv](https://github.com/astral-sh/uv)** for environment and dependency management. Python 3.13 required.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv
uv sync                                            # sync venv from pyproject.toml + uv.lock
```

---

## Other Commands

```bash
# Full pipeline by stage
uv run python -m pipeline.run_all         # ingest → clean → feature → train → predict → simulate
uv run python pipeline/05_inference.py    # one stage

# Serve directly without make demo
uv run uvicorn app.api.main:app --reload  # → http://localhost:8000/heatmap

# MLflow UI
uv run mlflow ui                          # → http://localhost:5000

# Tests + quality gates
uv run pytest                             # full suite
uv run ruff check app/ tests/
uv run mypy app/ tests/

# Airflow (PBI 25a skeleton)
cp orchestration/.env.example orchestration/.env
make airflow-init && make airflow-up      # → http://localhost:8080
```

---

## Future Work (designed, not built — deliberate scope)

Each of the demo modes below is **designed but unbuilt**. The design itself is the engineering signal; building it on a static historical dataset would be theatre.

### Closed training loop — monthly retrain DAG

The champion-vs-challenger gate (PBI 20) and inference fallback (PBI 21) already function as library calls — the static demo exercises them. Wrapping them in `train_monthly_dag` (PBI 25c) would add monthly + manual triggers and surface the gate decision in the Airflow UI; it would not add new ML capability. Skipping this DAG is a scope decision, not a gap.

### Closed feedback loop — drift detection and auto-retrain

The prediction log (PBI 22) already records every prediction with the model version that served it; once actuals arrive, a join job builds a ground-truth dataset. The designed extensions:

- **Drift detector** (PBI 23, Evidently): wrap `DataDriftPreset` (feature drift) and `RegressionPreset` (model performance) over a rolling 7-day window of the joined frame.
- **Alert + retrain trigger** (PBI 24): emit a `retrain_required` marker file when drift or WAPE degradation exceeds threshold.
- **Monitor DAG** (PBI 25d): hourly run of the detector; `FileSensor` on the marker fires the training DAG.

Drift monitoring on a finite historical dataset is largely theatre — there is no real drift to detect on TLC parquets you've already seen. The honest move is to document the design (here) rather than ship the Evidently wrapper.

### Outcome harness — driver-agent simulation

A `SimulatedClock` proves the *system* runs. It does **not** prove that drivers following the heatmap are better off. That requires a closed-loop simulation with synthetic drivers (PBI 28):

| Policy | Behaviour |
|---|---|
| `HeatmapFollower` | calls `/heatmap`, drives to a top-K predicted zone |
| `LastWeekFollower` | baseline — drives to whatever was busy this hour last week |
| `Stationary` | control — never moves |

At each simulated 15-min tick: idle drivers pick a destination per their policy → en-route drivers travel → on arrival, the harness consults the historical trip table to see if a real trip actually started in that zone during that window → if yes and a driver is there, the driver gets the trip and is locked into its historical duration → repeat. Output is a per-policy trips-served / idle-time / total-fare table in `docs/simulation_results.md`.

This is the single piece of post-MVP work that converts "the model has lift on WAPE" into "drivers using it earn more dollars." If portfolio time allows exactly one more PBI after Live Cadence ships, this is the one.

### Personalised ranking (DiDi-style)

Promotes the heatmap (Uber model: same view for everyone) to ranked per-driver recommendations (DiDi model: top-N zones scored by predicted demand divided by travel cost). PBIs 5–8 in the tracker. Heatmap leaves interpretation to the driver; ranking centralises it but requires fleet-state tracking to avoid herding. Deferred.

### Latency SLO + Kubernetes deployment

`/heatmap` currently runs comfortably under 200ms on the static dataset; PBI 29 would lock that in as a CI guard with a `bench_heatmap_p95` benchmark. A Kubernetes deployment (Cloud Run for serving + GKE Job for training) is a natural target environment but is not in scope for a portfolio demonstration — the docker-compose orchestration scaffold (PBI 25a) is sufficient signal.

---

## Honest Limitations

- **Data publication lag.** NYC TLC publishes trip records on a 1–2 month lag. This repo treats the data as if it streamed in real time; the 15-minute refresh cadence is *simulated*, not live. A production deployment would need an upstream telemetry feed.
- **Single-agent equivalent.** The driver-agent harness (when shipped) gives all N synthetic drivers the same heatmap and draws trips from a fixed historical pool. It captures recall-style lift, but does **not** model demand-supply equilibration — if every real driver flocked to the same zone, the prediction would self-defeat. A supply-aware demand model is a separate research project.
- **Macro patterns only.** The model captures temporal + spatial regularities. It does not yet incorporate weather, events, or transit disruptions — the cases where intuition most often fails. Roadmap items.

---

## Documentation Index

- `docs/2026_05_31_portofolio_readiness.html` — current portfolio readiness assessment (this rewrite)
- `docs/system_design.html` — overall architecture
- `docs/design_t30_15min_refresh.html` — t+30 / 15-min refresh design (B-track rationale)
- `docs/phase6_smoke_test_report.html` — end-to-end smoke report
- `implementation_checklist/pbi_tracker.md` — PBI history, dependencies, priority order — restructured around the six demo modes above

---

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

---

Built for portfolio review — SmartNews · Google L4 MLE / SWE-ML. Scope decisions are deliberate; "Future Work" sections document what was *chosen* to defer.
