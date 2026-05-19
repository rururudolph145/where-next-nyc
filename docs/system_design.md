# System Design — NYC Taxi Demand Forecasting

> End-to-end MLOps architecture for hourly taxi demand forecasting, built as a personal portfolio project.

---

## 1. Context & Goals

### Purpose
Forecast NYC taxi demand at the **(pickup zone, hour)** grain for short-horizon windows (**h+3, h+6, h+12**), exposed via a FastAPI HTTP endpoint and (later) a React heatmap UI.

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
| Latency target (serving) | < 200ms p95 for `/predict` |
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
│   (Socrata → Bronze parquet)  ───► (Bronze → Silver parquet)        │
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
│                        Training Pipeline                            │
│   app/training/trainer.py                                           │
│   (Gold features → time-split → LGBMRegressor → models/model.pkl)   │
│                       │                                             │
│                       └──► MLflow (mlruns/, model name:             │
│                                    ny_taxi_demand_model)            │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                Inference & Serving                                  │
│   app/inference/predictor.py   ◄── batch eval (MAE / RMSE / R²)     │
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
- Aggregates Silver trip records to **(pickup_zone, datetime_hour)** grain.
- Adds lag features (e.g., demand at h-1, h-24, h-168), rolling means, calendar features, weather joins (if applicable).
- Output: `data/gold/` parquet partitioned by month, target = `trip_count`.

### 3.4 Training
- **`app/training/trainer.py`** — `Trainer` class:
  - Loads Gold features.
  - **Time-based split only** — never shuffle temporal data (`pipeline.MD` §6).
  - `feature_target_separator()` drops `trip_count` to split X/y.
  - Trains using model class from `MODELS` dict (currently `LGBMRegressor`).
  - Saves to `models/model.pkl`.
  - Logs params, metrics, artifacts to MLflow under model name `ny_taxi_demand_model`.

### 3.5 Inference (offline evaluation)
- **`app/inference/predictor.py`** — `Predictor` class:
  - Loads `models/model.pkl`.
  - `feature_target_separator()` (mirrors trainer).
  - Computes MAE / RMSE / R² on held-out time slice.
  - Used by batch eval scripts; not by the API.

### 3.6 Serving (online)
- **`app/api/main.py`** — FastAPI app, exposes `GET /heatmap`.
- Loads pre-computed predictions at **startup** (inference simulation must be run before server starts).
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
Socrata API ──► raw trips JSON
            ──► data/bronze/yyyy=YYYY/mm=MM/trips.parquet  (raw schema)
            ──► clean.py
            ──► data/silver/yyyy=YYYY/mm=MM/trips.parquet  (cleaned + time features)
```

### 4.2 Feature Engineering Flow
```
data/silver/**.parquet
   ──► groupby(pickup_zone, hour_bucket)
   ──► aggregate trip_count
   ──► join lag/rolling/calendar features
   ──► data/gold/yyyy=YYYY/mm=MM/features.parquet
       schema: [pickup_zone, datetime_hour, hour, day_of_week, month,
                is_weekend, lag_1h, lag_24h, lag_168h, roll_24h_mean, ...,
                trip_count]
```

### 4.3 Training Flow
```
data/gold/**.parquet
   ──► time-split (last N% by datetime_hour as test)
   ──► X_train, y_train, X_test, y_test
   ──► LGBMRegressor.fit(X_train, y_train)
   ──► MLflow.log_params + log_metrics + log_artifact(model.pkl)
   ──► models/model.pkl
```

### 4.4 Serving Interface — `POST /predict`

**Request**
```json
{
  "pickup_zone": 132,
  "datetime_hour": "2026-05-18T14:00:00Z",
  "horizons": ["h+3", "h+6", "h+12"]
}
```

**Response**
```json
{
  "pickup_zone": 132,
  "base_time": "2026-05-18T14:00:00Z",
  "predictions": [
    {"horizon": "h+3",  "target_time": "2026-05-18T17:00:00Z", "trip_count": 42.7},
    {"horizon": "h+6",  "target_time": "2026-05-18T20:00:00Z", "trip_count": 58.1},
    {"horizon": "h+12", "target_time": "2026-05-19T02:00:00Z", "trip_count": 18.3}
  ],
  "model_version": "ny_taxi_demand_model@1"
}
```

**Errors**
| Status | Cause |
|---|---|
| 400 | Invalid zone id, malformed datetime, unsupported horizon |
| 404 | No feature row available for the requested base time |
| 503 | Model not loaded at startup |

---

## 5. Key Design Decisions

| Decision | Rationale |
|---|---|
| **Medallion architecture** | Clear layering; Bronze = raw audit trail, Silver = clean, Gold = ML-ready. |
| **Parquet on local FS** | Simple, columnar, zero-infra. Good fit for portfolio scope. |
| **DVC for data versioning** | Reproducible pipelines without a data warehouse. |
| **LGBMRegressor first** | Strong tabular baseline, fast, handles missing values; deep models can be added later behind the `MODELS` dict. |
| **Time-based split only** | Avoids leakage on temporal data. Hard rule (`pipeline.MD` §6). |
| **MLflow local store** | No tracking-server infra needed; `mlruns/` is git-tracked-out. |
| **Model loaded at FastAPI startup** | Predictable cold-start cost paid once, sub-200ms p95 inference. |
| **`app/api/main.py` as canonical app** | Directly served via `uvicorn app.api.main:app` without re-export shims. |
| **`config.yml` over env vars for ML knobs** | Hyperparameters and paths belong in versioned config; `.env` is only for secrets. |

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
- Integration test: smoke-test `/predict` against a fixture model.
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
| Latency target | Bench `/predict` with `wrk` or `hey`; verify model is loaded once at startup |
| Reproducible data | DVC pins Bronze/Silver/Gold snapshots |
| No data committed | `.gitignore` covers `data/**/*.parquet` and `*.csv` |

---

## 9. Open Questions & Follow-Ups

- [ ] Decide on **weather features** join source (NOAA? open-meteo?).
- [ ] Pick a strategy for **multi-horizon forecasting**: separate models per horizon vs single model with horizon as feature.
- [ ] Add `/health` and `/model-info` endpoints.
- [ ] Wire **Evidently** drift reports into a scheduled job.
- [ ] Decide on heatmap rendering library for the frontend (deck.gl vs leaflet vs Mapbox).
- [ ] Containerize serving and write a minimal `Dockerfile`.

---

## 10. References
- `pipeline.MD` — pipeline-stage checklist (ingestion → monitoring).
- `Roadmap.MD` — feature roadmap.
- `project_checklist.MD` — implementation tracking.
- `CLAUDE.md` — repo-level configuration and rules.
