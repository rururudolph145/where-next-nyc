# MLOps Roadmap: NYC Taxi Demand Forecasting

**Timeline:** 6 weeks (~3 hours/day)
**Goal:** Productionize the MVP model to serve two personas with proper MLOps best practices.

---

## Personas

| | Fleet Owner (B2B) | Taxi Driver (B2C) |
|---|---|---|
| **Behavior** | Dashboard, batch queries, periodic decisions | Frequent checks, location-based, time-sensitive |
| **Traffic** | Low QPS, heavy queries | High QPS, lightweight requests |
| **Latency** | <2s acceptable | <200ms required |
| **Endpoints** | `POST /fleet/forecast`, `/fleet/heatmap`, `GET /fleet/history` | `GET /driver/demand`, `/driver/nearby`, `/driver/hotspots` |

---

## Week 1: Complete the Core Pipeline (Notebook → Production Code)

**Goal:** Reproducible end-to-end pipeline: `ingest → clean → feature → train → evaluate`

- [ ] Create `steps/feature.py` — `FeatureEngineer` class
  - [ ] `aggregate_hourly()`: floor pickup times, group by (hour, zone), count trips, fill zero-demand slots
  - [ ] `add_cyclical_time()`: sin/cos encoding for hour_of_day, day_of_week, month
  - [ ] `add_zone_coordinates()`: merge centroid lat/lon from reference shapefile
  - [ ] `add_lag_features()`: per-zone lags at t-1, t-3, t-6, t-12, t-24, t-168
  - [ ] `add_rolling_features()`: rolling mean/std over 6h, 24h, 168h windows per zone
  - [ ] `run()`: full Silver → Gold pipeline method
- [ ] Create `data/reference/zone_centroids.csv` from taxi zone shapefile
- [ ] Update `config.yml`
  - [ ] Add `lag_hours: [1, 3, 6, 12, 24, 168]`
  - [ ] Add `rolling_windows: [6, 24, 168]`
  - [ ] Fix `train.shuffle: false` (currently `true` — wrong for time-series)
- [ ] Fix `steps/train.py` — add type hints, Google docstrings, temporal split, expand MODELS dict
- [ ] Fix `steps/predict.py` — add type hints, Google docstrings
- [ ] Update `pipeline/run_all.py` — wire full pipeline: ingest → clean → feature → train → evaluate
- [ ] Fix `tests/test_clean.py` — broken import (`Cleaner` → `DataCleaner`)
- [ ] Create `tests/test_feature.py`
  - [ ] Test no future data leakage in lag features
  - [ ] Test cyclical encoding values in [-1, 1]
  - [ ] Test zero-fill produces complete zone×hour grid
- [ ] Create `tests/test_train.py`
  - [ ] Test temporal split preserves time ordering
  - [ ] Test model trains and prediction shape is correct
- [ ] Verify: `uv run python -m pipeline.run_all` produces `models/model.pkl`
- [ ] Verify: `uv run pytest` — all tests pass

---

## Week 2: MLflow Integration + Experiment Tracking

**Goal:** Every training run tracked, models versioned, promotion path defined.

- [ ] Integrate MLflow into `steps/train.py`
  - [ ] Log hyperparameters, feature list, training row count
  - [ ] Log model artifact with signature to registry (`ny_taxi_demand_model`)
  - [ ] Log training metrics (MAE on train set)
- [ ] Integrate MLflow into `steps/predict.py`
  - [ ] Log eval metrics: MAE, RMSE, R², WAPE to active run
  - [ ] Log per-zone error breakdown as artifact
- [ ] Create `steps/registry.py` — model registry utilities
  - [ ] `promote_best_model()`: find best run by test_mae, transition to Production
  - [ ] `load_production_model()`: load current Production model from registry
- [ ] Create `steps/tune.py` — hyperparameter tuning
  - [ ] Optuna search space for LGBMRegressor (n_estimators, lr, num_leaves, max_depth, subsample)
  - [ ] Temporal cross-validation (not random)
  - [ ] Each trial logged as MLflow child run
- [ ] Update `config.yml` — add `mlflow.experiment_name`, `mlflow.tracking_uri`
- [ ] Create `tests/test_registry.py` with mock MLflow
- [ ] Verify: `uv run python -m pipeline.run_all` → `uv run mlflow ui` shows runs and registered model

---

## Week 3: Dual API Design + Redis Caching

**Goal:** Two distinct API surfaces, pre-computed predictions cached in Redis.

- [ ] Create API package structure
  - [ ] `app/api/main.py` — FastAPI app factory with lifespan (load model, connect Redis)
  - [ ] `api/dependencies.py` — DI providers: model, Redis client, config
  - [ ] `api/models/requests.py` — Pydantic request schemas
  - [ ] `api/models/responses.py` — Pydantic response schemas
- [ ] Create Driver API (B2C) — `api/routers/driver.py`
  - [ ] `GET /api/v1/driver/demand?zone_id=161&horizon=3` — single-zone forecast
  - [ ] `GET /api/v1/driver/nearby?lat=..&lon=..&horizon=3&top_k=5` — nearest high-demand zones
  - [ ] `GET /api/v1/driver/hotspots?horizon=3&top_k=10` — top demand zones now
  - [ ] All reads from Redis cache (no live model.predict in request path)
- [ ] Create Fleet Owner API (B2B) — `api/routers/fleet.py`
  - [ ] `POST /api/v1/fleet/forecast` — multi-zone, multi-horizon batch forecast
  - [ ] `POST /api/v1/fleet/heatmap` — all-zone demand grid for dashboard
  - [ ] `GET  /api/v1/fleet/history?zone_id=161&days=30` — actual vs predicted history
- [ ] Create `api/routers/health.py` — `/health`, `/health/ready`, `/health/metrics`
- [ ] Create `api/services/prediction.py` — shared prediction logic
- [ ] Create `api/services/cache.py` — Redis caching layer (key: `pred:{zone}:{horizon}:{hour}`, TTL: 15 min)
- [ ] Create `jobs/precompute.py` — hourly batch job: predict all 263 zones at h+1..h+12, write to Redis (75-min TTL)
- [ ] Add `redis` to `pyproject.toml`
- [ ] Create `docker-compose.yml` — API + Redis for local dev
- [x] Deprecate old `app.py` (redirect imports or remove) — done in refactor/app-package-layout
- [ ] Create `tests/test_api_driver.py` — driver endpoint tests with TestClient + fakeredis
- [ ] Create `tests/test_api_fleet.py` — fleet endpoint tests
- [ ] Create `tests/test_cache.py` — cache layer unit tests
- [ ] Verify: `docker compose up` → both API surfaces respond correctly

---

## Week 4: Load Testing + Data Validation + Monitoring

**Goal:** Prove SLAs under load. Catch data quality issues. Detect model degradation.

### Load Testing (~1.5 hours/day)

- [ ] Add `locust` to `pyproject.toml`
- [ ] Create `loadtests/locustfile.py`
  - [ ] `DriverUser` class (weight=10, wait 1-5s): demand, nearby, hotspots tasks
  - [ ] `FleetUser` class (weight=1, wait 30-120s): forecast, heatmap tasks
- [ ] Create `loadtests/k6_driver.js` — spike test: ramp 50→200→500→0 users over 9 min
- [ ] Define and validate SLA targets:
  - [ ] `GET /driver/demand`: 500 QPS, p95 < 200ms, error < 0.1%
  - [ ] `GET /driver/nearby`: 200 QPS, p95 < 300ms, error < 0.1%
  - [ ] `POST /fleet/forecast`: 10 QPS, p95 < 2000ms, error < 0.5%
- [ ] Run load tests, identify bottlenecks, tune (connection pooling, async, etc.)
- [ ] Document load test results in `loadtests/RESULTS.md`

### Data Validation + Monitoring (~1.5 hours/day)

- [ ] Add `evidently` to `pyproject.toml` (if not already present)
- [ ] Create `steps/validate.py` — `DataValidator` class
  - [ ] `validate_bronze()`: schema check, null rates, expected columns
  - [ ] `validate_silver()`: distribution drift vs reference month (Evidently `DataDriftPreset`)
  - [ ] `validate_gold()`: feature completeness, no-leakage sanity check
- [ ] Wire validation gates into `pipeline/run_all.py` — halt pipeline on critical validation failures
- [ ] Create `monitoring/dashboard.py` — Evidently report generation
  - [ ] Prediction drift report (current vs reference predictions)
  - [ ] Accuracy report (predicted vs actual once actuals arrive)
  - [ ] Feature drift report (input distribution shift)
- [ ] Create `monitoring/alerts.py` — alert rule definitions
  - [ ] MAE degradation (warning > 25, critical > 50)
  - [ ] Data drift score > 0.3
  - [ ] Null rate spike > 5%
  - [ ] Prediction volume drop
- [ ] Enhance `api/routers/health.py` — `/health/metrics` returns cache hit rate, latency, model version
- [ ] Create `tests/test_validate.py`
- [ ] Verify: pipeline halts on intentionally corrupted data; drift report generates HTML

---

## Week 5: CI/CD + Containerization

**Goal:** Automated testing on every push. Docker-based deployment.

### CI Pipeline

- [ ] Create `.github/workflows/ci.yml`
  - [ ] Lint job: `ruff check .` + `ruff format --check .`
  - [ ] Typecheck job: `mypy . --strict`
  - [ ] Test job: `pytest --cov` (with Redis service container)
  - [ ] Integration test job: `pytest tests/integration/ -m integration`
- [ ] Create `tests/integration/test_pipeline_e2e.py` — small sample data through full pipeline
- [ ] Create `tests/smoke/test_health.py` — post-deploy health check
- [ ] Create `tests/smoke/test_predict_live.py` — post-deploy prediction sanity check

### CD Pipeline + Containers

- [ ] Create `.github/workflows/deploy.yml`
  - [ ] Build Docker image → push to registry
  - [ ] Deploy to staging → run smoke tests
  - [ ] Manual approval gate → deploy to production
- [ ] Fix `dockerfile` — upgrade Python 3.11→3.13, pip→uv
- [ ] Update `docker-compose.yml` — add MLflow service
- [ ] Update `makefile` with uv-based targets
  - [ ] `make test`, `make lint`, `make typecheck`
  - [ ] `make api`, `make load-test`, `make docker-up`, `make precompute`
- [ ] Verify: push to branch triggers CI; `docker compose up` starts full stack

---

## Week 6: Retraining Automation + Production Hardening

**Goal:** Self-maintaining system with automated retraining, prediction intervals, and cold-start handling.

### Retraining Automation

- [ ] Create `jobs/retrain.py` — `RetrainingOrchestrator`
  - [ ] Trigger conditions: scheduled (every 7 days) OR drift score > 0.3 OR rolling MAE > threshold
  - [ ] Run full pipeline with latest data
  - [ ] Champion/challenger comparison: promote only if challenger beats on overall MAE AND worst-zone MAE doesn't regress >10%
  - [ ] Archive retired model versions (never delete)
- [ ] Document rollback procedure in `docs/runbook.md`

### Prediction Intervals

- [ ] Update `steps/train.py` — train quantile regression models (q10, q50, q90) using LightGBM `objective='quantile'`
- [ ] Update `api/models/responses.py` — add `demand_low`, `demand_high`, `confidence` fields
- [ ] Update `jobs/precompute.py` — cache intervals alongside point predictions

### Cold-Start Handling

- [ ] Update `steps/feature.py` — `handle_cold_start()`
  - [ ] Zones with <24h history: use borough-level average
  - [ ] Zones with <168h history: fill missing lags with zone mean
  - [ ] Add `is_cold_start` boolean feature column

### Final Verification

- [ ] Full end-to-end test: API ingest → validate → feature → train → register → precompute → serve
- [ ] Load test passes all SLA targets
- [ ] CI pipeline green on main branch
- [ ] MLflow shows experiment history and production model
- [ ] Monitoring dashboard generates reports
- [ ] Update `pipeline.MD` checklist — check off all completed items
