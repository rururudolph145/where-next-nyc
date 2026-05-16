# PBI Tracker — NYC Taxi Demand Forecasting

> Organized by User Story Map activities. Start with Activity 1 & 2 (Walking Skeleton).
> Strategy: build the full pipeline end-to-end for single-horizon (time t) first,
> then expand to multi-horizon (t+30, t+60) — adding a new model is trivial
> since only the training target changes.
>
> **API approach:** Heatmap-style (Uber model) — serve all zone predictions and
> let the driver interpret. Personalized ranking (DiDi model) deferred to post-MVP.

## Release: MVP (PBIs 1–4)

### Activity 1: Intelligence & Forecasting (Single-Horizon)

Supports **Step 2 (Check Demand)** — get the core ML pipeline working first.

- [x] **PBI 1 — Single-Horizon Forecasting Engine (time t)**
  - Train model to predict trip_count at current time t per zone
  - Input: Gold layer features (time, zone, lag demand)
  - Output: predicted trip_count per zone at time t
  - Acceptance: MAE/RMSE within acceptable thresholds; MLflow-tracked experiment
  - Note: model and trainer already exist — this PBI validates the full loop

- [x] **PBI 2 — Inference Pipeline (time t)**
  - Build pipeline step that loads trained model and generates predictions from Gold data
  - Input: trained model + Gold layer features
  - Output: predictions table (zone_id, predicted_trip_count, timestamp)
  - Acceptance: pipeline script runs end-to-end; evaluation metrics logged

### Activity 2: Demand Heatmap API

Supports **Step 1 (Start Shift) → Step 2 (Check Demand)** in the driver journey.

- [x] **PBI 3 — Heatmap API**
  - Build `GET /heatmap` endpoint that reads the predictions table and returns all zone predictions
  - Input: none (reads latest predictions parquet)
  - Output: JSON with `generated_at` + list of `{ pulocationid, zone_name, predicted_trip_count }`
  - Acceptance: response < 200ms; returns valid zone predictions; serves from predictions store

### Activity 3: Multi-Horizon Upgrade

- [ ] **PBI 4 — Multi-Horizon Forecasting (t+30, t+60)**
  - Extend training to produce shifted targets (trip_count at t+30, t+60)
  - Only changes: target column generation in feature engineering + retrain
  - Inference pipeline structure stays identical
  - Input: Gold layer features with shifted targets
  - Output: predicted trip_count per zone per horizon
  - Acceptance: separate models or multi-output model; metrics logged per horizon

---

## Release: Post-MVP

### Personalized Ranking (upgrade from heatmap to recommendations)

> Requires zone mapping + distance-decay scoring. This is the DiDi-style approach
> where each driver gets a personalized ranked list instead of a shared heatmap.

- [ ] **PBI 5 — Zone Mapping Utility**
  - Develop geometric lookup to identify which NYC Taxi Zone a (lat, lon) falls in
  - Input: (lat, lon)
  - Output: taxi zone ID (integer)
  - Acceptance: correctly maps known coordinates to their zones; uses TLC zone shapefile

- [ ] **PBI 6 — User Location Capture**
  - Build logic to retrieve (lat, lon) from the driver's device (simulated or real)
  - Input: device GPS or simulated coordinates
  - Output: (lat, lon) tuple
  - Acceptance: returns valid NYC coordinates; handles out-of-bounds gracefully

- [ ] **PBI 7 — Distance-Decay Implementation**
  - Write the scoring function: `efficiency = predicted_demand / travel_distance`
  - Input: driver zone, target zone, predicted demand
  - Output: efficiency score per zone
  - Acceptance: score correctly penalizes far-away zones; unit tested

- [ ] **PBI 8 — Top-N Recommendation API**
  - Upgrade `/heatmap` or add `POST /recommend` endpoint that returns ranked zones by efficiency score
  - Input: POST with driver (lat, lon)
  - Output: JSON list of top-N zones with scores
  - Acceptance: response < 200ms; returns valid zone recommendations

### Near-Real-Time Inference (simulated streaming)

> In production, Uber refreshes heatmap predictions every 10-15 minutes.
> This doesn't require a streaming inference pipeline — prediction is batch by nature.
> The two loops are decoupled:
>
> - **Data ingestion** (continuous): trip events land in a database in near-real-time.
> - **Inference** (batch, every 15 min): cron job queries latest aggregated counts,
>   computes features, runs model.predict(), updates the predictions store.
>
> The model needs aggregates over completed time windows (e.g., `trip_count_lag_1` =
> total pickups in zone X one hour ago). You can't compute this until the window closes,
> so streaming individual trip events doesn't help — batch on aggregated windows is correct.
>
> **Simulation approach:** Since we can't run 24/7, we replay historical data with a
> simulated clock. A `--simulate-time` flag advances through historical hours, exercising
> the same code path as a production cron job. In a design doc, explain:
> "In production this would be triggered by a cron job every 15 minutes against a live
> database. For demonstration, we replay historical data with a simulated clock."

- [ ] **PBI 11 — Near-Real-Time Inference with Simulated Clock**
  - Add `--simulate-time` flag to inference pipeline that replays historical data hour-by-hour
  - Each iteration: query data "up to" simulated time, compute features, predict, update heatmap
  - Simulating a full month completes in minutes but exercises the production code path
  - Input: historical Gold data + simulated current time
  - Output: predictions table updated per simulated hour
  - Acceptance: heatmap updates as simulated clock advances; same results as full batch

### System Reliability

- [ ] **PBI 9 — Stale Data Handler**
  - Fallback to historical averages when live forecast service is unavailable
  - Input: failed forecast lookup
  - Output: historical average demand per zone
  - Acceptance: graceful degradation; driver always gets a recommendation

- [ ] **PBI 10 — Model Monitoring (L5 Signal)**
  - Log predicted vs actual pickups to detect model drift
  - Input: predictions table + actual trip data
  - Output: drift metrics logged to MLflow / monitoring dashboard
  - Acceptance: alert when prediction error exceeds threshold

---

## Dependency Graph

```
PBI 1 (train t) ──> PBI 2 (inference t) ──> PBI 3 (heatmap API)
                                  │
                                  ├──> PBI 4 (multi-horizon)
                                  │
                                  ├──> PBI 11 (simulated streaming) ──> PBI 3 (heatmap refreshes)
                                  │
                                  │         PBI 5 (zone map) ──┐
                                  │                             ├──> PBI 7 (decay) ──> PBI 8 (recommend API)
                                  │         PBI 6 (location) ──┘
                                  │
                                  ├──> PBI 9 (fallback)
                                  └──> PBI 10 (monitoring)
```

## Priority Order

| Order | PBI | Rationale |
|-------|-----|-----------|
| 1 | PBI 1 | Validate existing single-horizon model end-to-end |
| 2 | PBI 2 | Inference pipeline — completes the full ML loop |
| 3 | PBI 3 | Heatmap API — serves predictions to drivers (MVP complete) |
| 4 | PBI 4 | Multi-horizon — only changes training target, pipeline reused |
| 5 | PBI 5 | Zone lookup — needed by personalized ranking |
| 6 | PBI 6 | Location capture — simple, enables testing |
| 7 | PBI 7 | Scoring function — pure math, unit testable |
| 8 | PBI 8 | Recommendation API — personalized ranking endpoint |
| 9 | PBI 9 | Fallback — reliability layer |
| 10 | PBI 10 | Monitoring — observability layer |
| 11 | PBI 11 | Simulated streaming — shows production awareness |

## Known Limitations & Future Considerations

- **Feedback loop:** Driver recommendations influence future trip data (see DiDi's multi-driver optimization). Not addressed in MVP since heatmap approach + surge pricing mitigate herding naturally.
- **Multi-driver coordination:** MVP serves the same heatmap to all drivers (Uber model). DiDi-style per-driver optimization requires fleet state tracking and is deferred to post-MVP.
- **A/B testing:** No framework for measuring if recommendations improve driver outcomes. Design doc recommended before scaling.
- **Batch vs streaming inference:** Prediction is batch by nature — the model needs aggregates over completed time windows (e.g., hourly trip counts). Data ingestion can be streaming into a DB, but inference runs as a periodic batch job (every 15 min in Uber's case). These two loops are decoupled. For demo purposes, historical replay with a simulated clock exercises the same code path as a production cron job.
