# System Design — NYC Taxi Demand Forecasting

> End-to-end MLOps architecture for short-horizon NYC taxi demand forecasting, built as a personal portfolio project. Current default config (PBI B2): 15-minute aggregation bucket + 30-minute summed-horizon target with k-NN spatial smoothing.

---

## 1. Context & Goals

### Purpose
Forecast NYC taxi demand at the **(pickup zone, 15-min bucket)** grain. The default training target is `target_sum_t1_t2` — the sum of trip counts over the next two 15-min buckets, i.e. the cumulative 30-minute lookahead. Predictions are exposed via a FastAPI HTTP endpoint and (later) a React heatmap UI.

**Forecasting taxonomy** (see `pbi_tracker.md` Activity 3b):
- **Pure nowcast** (B1, target window `[t, t+30)` as one bin): kept in-tree, currently off by default.
- **Extended nowcast** (B2, target window `[t, t+30)` as sum of two 15-min bins): current default — predicts cumulative demand from the prediction time onward.
- **True forecast** (target window `[t+30, t+60)`, no overlap with `t`): supported by the `horizon_buckets` knob (PBI A3), currently set to 0.

### Portfolio Goals
- Demonstrate a clean **medallion architecture** (Bronze → Silver → Gold).
- Showcase **MLflow** experiment tracking and **DVC** data versioning.
- Provide a working **FastAPI** prediction service with production-quality code (strict typing, tests, lint/format).
- Lay groundwork for a frontend visualization layer.

### Non-Goals
- Production-grade scale (single-node, local-first).
- Multi-tenant auth (no user accounts, no rate limiting).
- Real-time streaming (batch hourly cadence is sufficient).

### Constraints
| Constraint | Value |
|---|---|
| Compute | Single-node, optional CUDA 12.6 |
| Storage | Local filesystem, parquet-based |
| Runtime | Python 3.13+, uv-managed |
| Latency target (serving) | < 200ms p95 for `GET /heatmap` |
| Refresh cadence (serving) | 15 minutes (production-equivalent cron); simulated via `--simulate-time` |
| Training cadence | Manual / nightly cron |
| Data volume | ~10–100M trip rows/year, downsampled per slice |

---

## 2. System Architecture (High-Level)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         External Sources                            │
│  ┌─────────────────────┐                ┌─────────────────────┐     │
│  │  NYC TLC Socrata    │                │   NYC Zone Lookup   │     │
│  │  Trip Records API   │                │   (taxi_zones.csv)  │     │
│  └──────────┬──────────┘                └──────────┬──────────┘     │
└─────────────┼────────────────────────────────────────┼──────────────┘
              │                                        │
              ▼                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Ingestion Pipeline                           │
│   app/ingestion/ingest.py          app/ingestion/clean.py           │
│   (raw parquet → Bronze)      ───► (Bronze → Silver)                │
│   (parallel reads, ThreadPoolExecutor; sort-once invariant)         │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
              ┌────────────────────────────┐
              │   Medallion Data Lake      │
              │   data/bronze/             │
              │   data/silver/             │
              │   data/gold/  (features)   │
              │   (DVC-versioned)          │
              └────────────┬───────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Feature Engineering (Silver → Gold)                    │
│   app/ingestion/feature.py — FeatureEngineer.run():                 │
│     build_hourly_demand   (per-trip → (zone, bucket) panel)         │
│     add_time_features     (hour/dow/month + cyclical encodings)     │
│     add_lag_features      (lags in bucket units, PBI A2)            │
│     add_shifted_target    (single future bucket, PBI A3; off=0)     │
│     add_summed_horizon_target (sum next N buckets, PBI B2)          │
│     add_zone_coordinates  (centroid join, drops orphan zones)       │
│     add_neighbor_lag_smoothing (k-NN mean of lag_1, PBI B1)         │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                            │
│   app/training/trainer.py                                           │
│   (Gold features → time-split → LGBMRegressor → models/model.pkl)   │
│   target = train.target (default: target_sum_t1_t2)                 │
│                       │                                             │
│                       └──► MLflow (mlruns/, experiment:             │
│                                    ny_taxi_demand_model)            │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                Inference & Serving                                  │
│   app/inference/predictor.py   ◄── batch eval (MAE / RMSE / R²)     │
│   app/inference/simulator.py    ── hourly replay; writes            │
│                                    data/predictions/*.parquet       │
│   app/api/main.py               ── FastAPI GET /heatmap             │
│   (uvicorn app.api.main:app)                                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                Consumers (planned)                                  │
│   React + Vite frontend (heatmap) ── curl / Swagger UI              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Responsibilities

### 3.1 Configuration
- **`app/core/config.py`** — single source of YAML config loading via `load_config()`.
- **`config.yml`** — owns: data paths (bronze/silver/gold), train/test split, model hyperparameters, MLflow experiment name.

### 3.2 Ingestion (Bronze → Silver)
| Module | Responsibility |
|---|---|
| `app/ingestion/ingest.py` | `Ingestion` class. Downloads raw trip data from the Socrata API into `data.bronze_path`. Month-by-month iteration. |
| `app/ingestion/clean.py` | `DataCleaner` class. Drops nulls, removes non-positive trip values, clips fare outliers via IQR, engineers time features (`hour`, `day_of_week`, `month`, `is_weekend`). |
| `pipeline/run_all.py` | Orchestrates ingest → clean → feature → train → infer month-by-month. |

### 3.3 Feature Engineering (Silver → Gold)
- **`app/ingestion/feature.py`** — `FeatureEngineer.run()` is the canonical pipeline. Configuration-driven via `feature.*` keys.
- Aggregates Silver trip records to a **(pickup_zone, bucket)** grain using `dt.floor(self.bucket)`. Default `bucket = "15min"`.
- Stage order (each method idempotent against its config knobs):
  1. `build_hourly_demand` — groupby → `trip_count`. (The "hourly" name is historical; bucket width comes from `self.bucket`.)
  2. `add_time_features` — `hour`, `day_of_week`, `month` + sin/cos cyclical encodings.
  3. `add_lag_features` — emits `trip_count_lag_{n}` for each `n` in `feature.lags` (bucket offsets, **not** hours; PBI A2).
  4. `add_shifted_target` (PBI A3) — emits `target_t{N}` = `groupby(zone).shift(-N)` of `trip_count`. **Off by default** (`horizon_buckets = 0`). True-forecast path.
  5. `add_summed_horizon_target` (PBI B2) — emits `target_sum_t1_t{N}` = sum of `groupby(zone).shift(-k)` for k=1..N. **On by default** with N=2 → 30-min summed lookahead. Extended-nowcast path.
  6. `add_zone_coordinates` — left-join centroid lat/lon from `taxi_zones.shp`; drops zones missing from the shapefile (typically 57, 264, 265).
  7. `add_neighbor_lag_smoothing` (PBI B1) — `sklearn.neighbors.BallTree` with `metric="haversine"` over zone centroids; for each `(zone, bucket)` row, computes mean of `trip_count_lag_1` across k nearest neighbours (self excluded). Emits `trip_count_lag_1_neighbor_mean`. `n_neighbors = 0` disables.
- Output: `data/gold/yellow_tripdata/...parquet` partitioned by month. Target column selected by `train.target`.

### 3.4 Training
- **`app/training/trainer.py`** — `Trainer` class:
  - Loads Gold features.
  - **Time-based split only** — never shuffle temporal data; ratios from `train.split_ratios`.
  - Features picked by name from `train.features`; target picked by name from `train.target`.
  - Trains using model class from `MODELS` dict (currently `LGBMRegressor` with L2 loss).
  - Saves to `models/model.pkl`.
  - Logs params, metrics, artifacts to MLflow under experiment `ny_taxi_demand_model`.

### 3.5 Inference (offline evaluation + simulation)
- **`app/inference/predictor.py`** — `Predictor` class:
  - Loads `models/model.pkl`.
  - Computes MAE / RMSE / R² on the held-out time slice.
  - Used by batch eval scripts; not by the API.
- **`app/inference/simulator.py`** — bucket-by-bucket replay over `inference.simulation.{start_hour, end_hour}` exercising the production inference code path against historical data. Output is a single predictions parquet at `inference.output_path`.

### 3.6 Serving (online)
- **`app/api/main.py`** — FastAPI app, exposes `GET /heatmap`.
- Reads the predictions parquet (`serving.predictions_path`) and the zone shapefile (`serving.zone_shapefile_path`) at startup; the simulator must have run first.
- Response: `{ generated_at, predictions: [{ pulocationid, zone_name, predicted_trip_count }, ...] }`. Pagination (`?top_n=`) is tracked under PBI 16 — currently returns all zones.
- Served via `uvicorn app.api.main:app`.

### 3.7 Frontend (planned)
- Vite + React + TypeScript SPA at `frontend/`.
- Uses **@tanstack/react-query** for FastAPI calls.
- Vite dev server proxies `/api` → `http://127.0.0.1:8000`.
- Primary view: zone × hour heatmap.

---

## 4. Data Flow & Interfaces

### 4.1 Ingestion Flow (per month)
```
Yellow Taxi parquet files (TLC)
            ──► load_raw (ThreadPoolExecutor, parallel reads, sort-once)
            ──► data/bronze/yellow_tripdata/bronze.parquet
            ──► clean.py (drop nulls, IQR outlier clip)
            ──► data/silver/yellow_tripdata_cleaned.parquet
```

### 4.2 Feature Engineering Flow (B2 default)
```
data/silver/yellow_tripdata_cleaned.parquet
   ──► dt.floor("15min")  ──► groupby(pulocationid, hour) → trip_count
   ──► add time features (hour/dow/month + sin/cos)
   ──► add lags in 15-min units: lag_{1,2,4,8,12,24,96,672}
        = 15min, 30min, 1h, 2h, 3h, 6h, 1d, 1w
   ──► add_summed_horizon_target(N=2) → target_sum_t1_t2
   ──► add zone centroids; drop orphan zones (57, 264, 265)
   ──► add_neighbor_lag_smoothing(k=5) → trip_count_lag_1_neighbor_mean
   ──► data/gold/yellow_tripdata/yyyy=YYYY/mm=MM/features.parquet
       schema: [pulocationid, hour,
                hour_of_day_{sin,cos}, day_of_week_{sin,cos}, month_{sin,cos},
                pickup_longitude, pickup_latitude,
                trip_count,
                trip_count_lag_{1,2,4,8,12,24,96,672},
                trip_count_lag_1_neighbor_mean,
                target_sum_t1_t2]
```

### 4.3 Training Flow
```
data/gold/**.parquet
   ──► time-split by row order (train/val/test = 0.8/0.1/0.1)
   ──► X = train.features (15 columns), y = train.target (target_sum_t1_t2)
   ──► LGBMRegressor.fit(X_train, y_train)
   ──► MLflow.log_params + log_metrics + log_artifact(model.pkl)
   ──► models/model.pkl
```

### 4.4 Serving Interface — `GET /heatmap`

Read-only endpoint backed by a pre-computed predictions parquet. The simulator (or, in production, a 15-min cron job) refreshes the parquet; the API just serves the latest snapshot.

**Request** — no parameters today. PBI 16 (deferred) adds `?top_n=<int>`.

**Response**
```json
{
  "generated_at": "2026-05-23T14:00:00Z",
  "predictions": [
    {"pulocationid": 132, "zone_name": "JFK Airport",  "predicted_trip_count": 87.3},
    {"pulocationid": 161, "zone_name": "Midtown Center", "predicted_trip_count": 65.1},
    ...
  ]
}
```

Each `predicted_trip_count` is the model's estimate for `target_sum_t1_t2` at the most recent feature row — i.e. cumulative pickups expected over the next 30 minutes.

**Errors**
| Status | Cause |
|---|---|
| 503 | Predictions parquet or zone shapefile missing at startup |

---

## 5. Key Design Decisions

| Decision | Rationale |
|---|---|
| **Medallion architecture** | Clear layering; Bronze = raw audit trail, Silver = clean, Gold = ML-ready. |
| **Parquet on local FS** | Simple, columnar, zero-infra. Good fit for portfolio scope. |
| **DVC for data versioning** | Reproducible pipelines without a data warehouse. |
| **LGBMRegressor with L2 loss** | Strong tabular baseline, fast, handles missing values; deep models can be added later behind the `MODELS` dict. L2 was preferred over Poisson because the summed-horizon target (B2) is a sum of bin counts — its tail is heavier than Poisson and L2 + count features are robust without tuning a link function. |
| **15-min aggregation bucket (B2)** | Sparser than hourly buckets (σ/μ scales like 1/√λ), but tractable when paired with spatial smoothing. Matches Uber's production 15-min heatmap refresh cadence. Bucket width is config-driven via `feature.bucket` — A1 generalised this. |
| **Summed-horizon target (B2)** | The driver-facing question is "how many pickups over the next 30 minutes?" — a sum, not a point forecast. Implemented as `sum_{k=1..N} groupby(zone).shift(-k)`, drops the trailing N rows per zone. Independent of the single-bucket horizon (A3), so both columns can coexist. |
| **k-NN spatial smoothing (B1)** | 263 TLC zones are too fine for raw counts to be stable (most zones see 0-3 pickups per 15-min bucket). Rather than coarsen space (Pranshu's k-means approach), preserve zones and add `trip_count_lag_1_neighbor_mean` — mean (not sum, to keep the scale) of lag_1 across k=5 nearest zones via haversine BallTree. Self-excluded to avoid trivial leakage. |
| **Time-based split only** | Avoids leakage on temporal data. Hard rule enforced by `Trainer.train()` — never `shuffle=True`. |
| **MLflow local store** | No tracking-server infra needed; `mlruns/` is git-ignored. |
| **Predictions parquet served at startup** | API is a thin read layer; refresh cadence belongs to the upstream simulator/cron, not the request path. Keeps `/heatmap` sub-200ms regardless of model size. |
| **`app/api/main.py` as canonical app** | Directly served via `uvicorn app.api.main:app` without re-export shims. |
| **`config.yml` over env vars for ML knobs** | Hyperparameters and paths belong in versioned config; `.env` is only for secrets. |
| **Parallel `load_raw` (PBI 14)** | `ThreadPoolExecutor.map` over parquet reads with a post-concat `sort_values("tpep_pickup_datetime")`. Sort-once invariant is locked by `test_bronze_is_temporally_sorted` (PBI 12) and `test_load_raw_parallel_equivalence` (PBI 13). |

---

## 6. Cross-Cutting Concerns

### Configuration
- All ML/data knobs live in `config.yml`, loaded once via `app.core.config.load_config()`.
- Secrets (Socrata app token, DB creds) live in `.env`, never committed.

### Logging
- Structured logs to stdout. Use Python's `logging` with module-level loggers.
- Log level controlled via env var; default INFO.

### Error Handling
- Boundary validation: Pydantic models on FastAPI request schema.
- Internal exceptions wrapped at the API layer into 4xx/5xx — never leak stack traces.
- Trainer/predictor raise specific exceptions; no bare `except`.

### Testing
- `tests/` mirrors `app/`.
- Unit tests for `clean`, `feature_engineering`, `trainer`, `predictor`.
- Integration test: smoke-test `/heatmap` against a fixture predictions parquet (`tests/test_serving.py`); end-to-end Silver→Gold→Train against synthetic data (`tests/test_e2e_synthetic.py`).
- All tests run via `uv run pytest`; CI enforces strict mypy + ruff.

### Observability
- MLflow for model lineage (params/metrics/artifacts).
- **Evidently** (planned) for drift monitoring against a reference dataset.
- `/health` endpoint (planned) for liveness/readiness.

### Security & Hygiene
- No secrets in code or `config.yml`.
- `.parquet` / `.csv` under `data/` are gitignored.
- CORS configured restrictively for the planned frontend dev origin only.

---

## 7. Deployment View

### Current (local / portfolio)
```
┌──────────────────────────────────────────────────────────┐
│  Developer machine (WSL2 / Linux)                        │
│                                                          │
│  uv run python -m pipeline.run_all       # pipeline      │
│  uv run python -m app.training.trainer   # training      │
│  uv run uvicorn app.api.main:app --reload # serving :8000│
│  uv run mlflow ui                 # mlflow UI :5000      │
└──────────────────────────────────────────────────────────┘
```

### Future
- Containerize `serving/` only (training stays offline).
- Optionally publish FastAPI container to a small VM or container service.
- Frontend served statically (Vite build → `dist/`).

---

## 8. Validation Against Constraints

| Constraint | Validated by |
|---|---|
| Python 3.13+, uv-managed | `pyproject.toml`, `.python-version`, uv lockfile |
| Strict typing | `uv run mypy app/ --strict` (must pass) |
| Lint/format | `uv run ruff check app/ && uv run ruff format --check app/` |
| Time-split discipline | Code review of `Trainer.train()` — never `shuffle=True` |
| Latency target | Bench `/heatmap` with `wrk` or `hey`; verify predictions parquet is loaded once at startup |
| Reproducible data | DVC pins Bronze/Silver/Gold snapshots |
| No data committed | `.gitignore` covers `data/**/*.parquet` and `*.csv` |

---

## 9. Open Questions & Follow-Ups

- [ ] Decide on **weather features** join source (NOAA? open-meteo?).
- [ ] If/when product asks for true forecasting (target window `[t+30, t+60)`), flip `horizon_buckets` on alongside `horizon_sum_buckets` — both columns coexist and the trainer picks one via `train.target`.
- [ ] Add `/health` and `/model-info` endpoints.
- [ ] Land PBI 16 (paginate `/heatmap` with `?top_n=`) — currently returns all 263 zones.
- [ ] Wire **Evidently** drift reports into a scheduled job.
- [ ] Decide on heatmap rendering library for the frontend (deck.gl vs leaflet vs Mapbox).
- [ ] Containerize serving and write a minimal `Dockerfile`.

---

## 10. References
- `implementation_checklist/pbi_tracker.md` — PBI status, Activity 3b (B0/B1/B2) design notes, dependency graph.
- `docs/design_t30_15min_refresh.html` — original B2 design memo (15-min refresh + 30-min summed-horizon target).
- `docs/phase6_smoke_test_report.html` — performance-rules audit that motivated PBIs 12–17.
- `README.md` — setup, run commands, configuration reference.
- `CLAUDE.md` — repo-level configuration and rules.
