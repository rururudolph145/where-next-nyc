# NYT Prediction Project — MVP Checklist

## System Design

### MVP Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE (offline, run once)             │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │ Socrata  │───>│  Bronze  │───>│  Silver  │───>│    Gold      │  │
│  │   API    │    │ (raw     │    │ (cleaned │    │ (features +  │  │
│  │          │    │  parquet) │    │  data)   │    │  target)     │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────┬───────┘  │
│   01_ingest.py    02_clean.py    03_feature.py          │          │
│                                                         v          │
│                                                  ┌──────────────┐  │
│                                                  │   Trainer    │  │
│                                                  │  (LGBMReg)  │  │
│                                                  └──────┬───────┘  │
│                                                         │          │
│                                          ┌──────────────┼────────┐ │
│                                          v              v        │ │
│                                   ┌───────────┐  ┌───────────┐  │ │
│                                   │model.pkl  │  │  MLflow   │  │ │
│                                   └─────┬─────┘  │  (metrics │  │ │
│                                         │        │   + model) │  │ │
│                                         │        └───────────┘  │ │
│                                  04_train.py                    │ │
└─────────────────────────────────────────┼───────────────────────┘ │
                                          │                         │
┌─────────────────────────────────────────┼─────────────────────────┘
│                 INFERENCE PIPELINE (batch, run once)                │
│                                          │                         │
│  ┌──────────┐                    ┌───────v──────┐                  │
│  │   Gold   │───────────────────>│  Predictor   │                  │
│  │  (data)  │                    │              │                  │
│  └──────────┘                    └───────┬──────┘                  │
│                                          │                         │
│                                          v                         │
│                                  ┌──────────────┐                  │
│                                  │ Predictions  │                  │
│                                  │  (.parquet)  │                  │
│                                  └───────┬──────┘                  │
│                                   05_inference.py                  │
└──────────────────────────────────────────┼─────────────────────────┘
                                           │
┌──────────────────────────────────────────┼─────────────────────────┐
│                       SERVING (online)   │                         │
│                                          v                         │
│  ┌────────┐    GET /heatmap     ┌──────────────┐                   │
│  │ Driver │ ──────────────────> │   FastAPI    │                   │
│  │  App   │ <────────────────── │   (reads     │                   │
│  │        │   JSON response     │  predictions │                   │
│  └────────┘                     │   parquet)   │                   │
│                                 └──────────────┘                   │
│                                                                    │
│  Response:                                                         │
│  { generated_at, predictions: [{ pulocationid, predicted_count }] }│
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Production Architecture (what changes at scale)

```
┌──────────────────────────────────────────────────────────────────────┐
│              DATA INGESTION (continuous, streaming)                   │
│                                                                      │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────────┐   │
│  │ Trip     │───>│  Message     │───>│  Database                 │   │
│  │ events   │    │  queue       │    │  (append-only trip store) │   │
│  │ (real-   │    │  (Kafka/     │    │                           │   │
│  │  time)   │    │   Redis)     │    │                           │   │
│  └──────────┘    └──────────────┘    └─────────────┬─────────────┘   │
│                                                    │                 │
└────────────────────────────────────────────────────┼─────────────────┘
                                                     │
┌────────────────────────────────────────────────────┼─────────────────┐
│              INFERENCE (batch, every 15 min via cron)                 │
│                                                    │                 │
│  ┌─────────────────────┐                   ┌───────v──────┐          │
│  │ Query latest hourly │──────────────────>│  Predictor   │          │
│  │ counts per zone     │  feature vector   │  (model.pkl) │          │
│  │ (SQL aggregation)   │  per zone         │              │          │
│  └─────────────────────┘                   └───────┬──────┘          │
│                                                    │                 │
│  No Bronze→Silver→Gold pipeline needed.            v                 │
│  Just query DB for aggregated counts       ┌──────────────┐          │
│  (lag_1h, lag_3h, ...) and time features.  │ Predictions  │          │
│                                            │   store      │          │
│                                            └───────┬──────┘          │
│                                                    │                 │
└────────────────────────────────────────────────────┼─────────────────┘
                                                     │
┌────────────────────────────────────────────────────┼─────────────────┐
│              SERVING (online, reads latest predictions)               │
│                                                    v                 │
│  ┌────────┐    GET /heatmap       ┌──────────────────┐               │
│  │ Driver │ ────────────────────> │    FastAPI        │               │
│  │  App   │ <──────────────────── │ (cache layer +   │               │
│  │        │    JSON response      │  predictions     │               │
│  └────────┘                       │  store)          │               │
│                                   └──────────────────┘               │
│                                                                      │
│  Heatmap refreshes every 15 min.                                     │
│  Same heatmap served to all drivers (Uber model).                    │
│  Personalized ranking (DiDi model) would add POST /recommend.        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

MODEL RETRAINING (separate loop, days/weeks)
  New historical data ──> full Bronze→Silver→Gold pipeline
  ──> retrain model ──> new model.pkl ──> Predictor picks up on next run
```

### Key Design Decisions

| Decision | MVP | Production |
|----------|-----|------------|
| Data ingestion | Batch file download from Socrata | Streaming events into DB |
| Feature computation | Full Bronze→Silver→Gold pipeline | SQL query for latest aggregates |
| Inference trigger | Manual (run script) | Cron every 15 min |
| Inference scope | All Gold data (historical) | All zones, latest features only |
| Heatmap freshness | Static (one-time run) | Refreshes every 15 min |
| Model retraining | Manual | Scheduled (days/weeks) |
| Personalization | None (shared heatmap) | Post-MVP: per-driver ranking |

---

## Golden Layer Structure

- [ ] Model monitoring schema: data, actual demand, predicted demand, predictive model

## Training Pipeline

- [x] Data ingestion from API → store to Bronze
- [x] Data cleaning from Bronze: outlier removal (abnormal zone, abnormal trip), time format enforcement → store to Silver
- [x] Data preprocessing from Silver: separation of time, cyclical feature for time, longitude/latitude feature
- Training prediction model
  - [x] Single-horizon model (trip_count at time t) with LGBMRegressor
  - [x] Time-based 80/10/10 split, evaluation (MAE/RMSE/MAPE/WAPE), MLflow logging
  - [ ] Multi-horizon forecasting: t+30 min prediction *(PBI 4)*
  - [ ] Multi-horizon forecasting: t+60 min prediction *(PBI 4)*
- [x] Save model

## Inference Pipeline

- [x] Load trained model + Gold data → generate batch predictions *(PBI 2)*
- [x] Build predictions table (pulocationid, hour, predicted_trip_count, actual_trip_count) *(PBI 2)*
- [x] Evaluate with MAE/RMSE/MAPE/WAPE *(PBI 2)*
- [x] Save predictions to parquet *(PBI 2)*

## Serving — Heatmap API

- [ ] `GET /heatmap` — read predictions parquet, return all zone predictions as JSON *(PBI 3)*

---

## User Story Map — PBI Breakdown

> See full details: `implementation_checklist/pbi_tracker.md`

### MVP

| PBI | Name | Status |
|-----|------|--------|
| 1 | Single-Horizon Forecasting Engine | [x] |
| 2 | Inference Pipeline | [x] |
| 3 | Heatmap API | [ ] |
| 4 | Multi-Horizon Forecasting (t+30, t+60) | [ ] |

### Post-MVP

| PBI | Name | Status |
|-----|------|--------|
| 5 | Zone Mapping Utility | [ ] |
| 6 | User Location Capture | [ ] |
| 7 | Distance-Decay Scoring | [ ] |
| 8 | Top-N Recommendation API | [ ] |
| 9 | Stale Data Handler | [ ] |
| 10 | Model Monitoring | [ ] |

---

## Portfolio Signal Checklist

What this project demonstrates to recruiters, and what to prioritize next.

### Already demonstrated

- [x] **ML Engineering** — End-to-end pipeline (ingest → train → inference), proper time-based split, evaluation metrics (MAE/RMSE/MAPE/WAPE)
- [x] **Data Engineering** — Medallion architecture (Bronze/Silver/Gold), parquet storage, config-driven pipeline
- [x] **Software Engineering** — Modular architecture, mypy strict, ruff linting, pytest coverage, Google-style docstrings
- [x] **MLOps awareness** — MLflow experiment tracking, model versioning, reproducible experiments

### High-impact items to complete

- [ ] **Product thinking: Heatmap API** *(PBI 3)* — Turns this from "ML pipeline" into "ML product". A working endpoint that a driver app could call.
- [ ] **Product thinking: Design doc** — Write a 1-2 page doc covering:
  - Why heatmap (Uber model) over personalized ranking (DiDi model) for MVP
  - The feedback loop problem: recommendations influence future training data
  - Multi-driver coordination: why naive recommendations cause herding
  - How you'd A/B test whether recommendations actually help drivers
- [ ] **System design: Scale considerations** — Document what changes at scale:
  - Batch refresh frequency (how often to re-run inference)
  - Caching layer between predictions store and API
  - Multi-driver coordination to prevent zone oversaturation
- [ ] **Communication: README** — Project story in 30-second-scannable format:
  - Problem → Approach → Architecture diagram → Sample API response → Results → Limitations → Next steps

### Why this ordering matters

> Model accuracy is the **least** differentiating part — everyone can train an LGBM.
> The pipeline, working API, and design reasoning are what set a candidate apart.
> Recruiters spend ~30 seconds on a repo; the README and architecture are what get clicked first.
