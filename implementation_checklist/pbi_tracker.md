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

## Release: Performance Hardening (PBIs 12–17)

> Aligns the codebase with `.claude/rules/performance.md`. Audit findings recorded in
> the Phase 6 smoke-test follow-up: `app/ingestion/ingest.py:load_raw` reads parquet
> files sequentially despite the "parallelize I/O-bound operations with async" rule;
> `pd.concat` accumulates all frames in memory in violation of "use generators when
> processing large lists"; `GET /heatmap` returns the full 263-zone payload in
> violation of "Use pagination when API responses are large".
>
> **Critical invariant:** Bronze data must remain globally sorted by
> `tpep_pickup_datetime` regardless of file read order. Parallel I/O is safe *only* if
> a stable sort runs after `pd.concat` and a regression test enforces it.

### Activity 4: Ingestion I/O Parallelism

Supports the "parallelize I/O-bound operations with async" rule. Splits cleanly into
test-first invariant locking → equivalence test → implementation, per TDD rule.

- [x] **PBI 12 — Lock-in Bronze Temporal-Sort Invariant (Test-First)** — *shipped in `b3ba4d9`*
  - Add `test_bronze_is_temporally_sorted` to `tests/test_dataset.py`
  - Assertion: `pd.read_parquet(bronze_path)["tpep_pickup_datetime"].is_monotonic_increasing`
  - Input: small fixture of in-order daily parquets (4 daily files across 2 months, each internally time-sorted)
  - Output: passing test against current sequential implementation
  - Acceptance: test passes on current code; pins the invariant before any parallelization touches the read path
  - Outcome: pytest baseline grew 57 → 58 passing; ruff + mypy clean on `app/ tests/`; the invariant currently holds by coincidence of the sequential loop — PBI 14 will be forced to add an explicit `sort_values` to keep it green

- [x] **PBI 13 — Parallel-Equivalence Regression Test** — *shipped (xfail-tracked) in a follow-up commit*
  - Add `test_load_raw_parallel_equivalence` to `tests/test_dataset.py`
  - Approach: outcome-based contract test rather than `max_workers=1` vs `max_workers=8` API check — the fixture arranges raw files so that **glob/concat order ≠ temporal order**, exercising the exact hazard parallel reads will introduce
  - Assertion: Bronze output is globally monotonic on `tpep_pickup_datetime` AND equals the canonical sorted reference
  - Marked `@pytest.mark.xfail(strict=True)` so the suite stays GREEN until PBI 14 lands; `strict=True` ensures the test will *fail* (xpass) the moment PBI 14 introduces the post-concat sort, providing automatic signal that PBI 14 is in
  - Acceptance: confirmed RED today (Bronze comes out `[Jan 15 × 4, Jan 5 × 4]`, not monotonic) — codifies the order-independence contract for PBI 14

- [ ] **PBI 14 — Parallelize `load_raw` with `ThreadPoolExecutor`**
  - Replace the sequential `for pq_file in parquets` loop in `app/ingestion/ingest.py` with `ThreadPoolExecutor.map(pd.read_parquet, paths)`
  - Add `ingest.max_workers` (default 8) to `config.yml`; thread the value through `Ingestion.__init__`
  - After `pd.concat`, call `df.sort_values("tpep_pickup_datetime", kind="mergesort").reset_index(drop=True)` to restore canonical time order
  - Acceptance: PBIs 12 + 13 pass; all 58 existing tests still pass; ingestion runtime measurably improves (informal benchmark in PR description)

- [ ] **PBI 15 — Streaming Bronze Write (Generator Compliance) — *stretch***
  - Replace `frames.append` accumulator + `pd.concat` with streaming `pyarrow.parquet.ParquetWriter`
  - Each parallel read writes its sorted batch incrementally; final external merge-sort pass produces canonically-ordered bronze
  - Honors the "use generators when processing large lists" rule
  - Acceptance: peak RSS during ingestion stays bounded regardless of date range; PBIs 12 + 13 still pass
  - Note: defer until ingestion actually OOMs in practice; YAGNI applies

### Activity 5: API Response Compliance

Supports the "Use pagination when API responses are large" rule.

- [ ] **PBI 16 — Paginate `GET /heatmap`**
  - Add `?top_n=<int>` query parameter to `app/api/main.py:get_heatmap` (default 50, max 263)
  - Sort predictions descending by `predicted_trip_count` (already done), then slice to top-N before serializing
  - Add `test_heatmap_top_n_default`, `test_heatmap_top_n_custom`, `test_heatmap_top_n_out_of_range` to `tests/test_serving.py`
  - Update `docs/system_design.md` §4.4 and `docs/system_design.html` to document the new parameter
  - Acceptance: default response payload shrinks by ~80%; serving tests stay 100% green

### Activity 6: Outbound HTTP Compliance — *Forward-looking*

> N/A today — no outbound HTTP calls live in `app/`. Activates when the Socrata
> downloader (currently external) is migrated into the repo.

- [ ] **PBI 17 — Bring Socrata Downloader Under Performance Rules**
  - When the Socrata downloader is added to `app/ingestion/`:
    - [ ] Set request `timeout=180` seconds
    - [ ] Configure `max_retries=3–5` via `tenacity` or `urllib3.Retry`
    - [ ] Exponential backoff on HTTP 429 (`backoff_factor` ≥ 1.0)
    - [ ] Parallelize multi-month downloads with `asyncio.gather` + `httpx.AsyncClient`
    - [ ] Reuse a single client/session — honors connection pooling
  - Acceptance: downloader survives a forced 429 burst in tests; total wall-clock for an N-month range scales sub-linearly with N

### Activity 7: Documentation Alignment

- [ ] **Update `about_concurrency.md`** — once PBI 14 lands, mark §5 ("Applying Concurrency to `pipeline.run_all`") as implemented and link to the live code
- [ ] **Add "Performance" subsection to `docs/system_design.md`** — codify the parallel-read + sort-once invariant as an architectural guarantee, not just an implementation detail
- [ ] **Append a Performance Compliance row to the Phase 6 smoke-test HTML report** — track which rules are now met

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
PBI 1 (train t) ──> PBI 2 (inference t) ──> PBI 3 (heatmap API) ──> PBI 16 (paginate heatmap)
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

Performance Hardening (independent of feature PBIs):
PBI 12 (sort invariant test) ──> PBI 13 (parallel equivalence test) ──> PBI 14 (parallel load_raw)
                                                                              │
                                                                              └──> PBI 15 (streaming write, stretch)

PBI 17 (Socrata HTTP compliance) — blocked: requires downloader to migrate into app/ first
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
| 12 | PBI 12 | Bronze temporal-sort invariant test — TDD RED step; safe in isolation |
| 13 | PBI 13 | Parallel-equivalence regression test — RED for PBI 14 |
| 14 | PBI 14 | Parallel `load_raw` — closes the "parallelize I/O-bound" gap |
| 15 | PBI 16 | Paginate `/heatmap` — closes the "pagination for large responses" gap |
| 16 | PBI 15 | Streaming bronze write — defer until ingestion OOMs in practice |
| 17 | PBI 17 | Socrata HTTP compliance — blocked until downloader migrates into `app/` |

## Known Limitations & Future Considerations

- **Feedback loop:** Driver recommendations influence future trip data (see DiDi's multi-driver optimization). Not addressed in MVP since heatmap approach + surge pricing mitigate herding naturally.
- **Multi-driver coordination:** MVP serves the same heatmap to all drivers (Uber model). DiDi-style per-driver optimization requires fleet state tracking and is deferred to post-MVP.
- **A/B testing:** No framework for measuring if recommendations improve driver outcomes. Design doc recommended before scaling.
- **Batch vs streaming inference:** Prediction is batch by nature — the model needs aggregates over completed time windows (e.g., hourly trip counts). Data ingestion can be streaming into a DB, but inference runs as a periodic batch job (every 15 min in Uber's case). These two loops are decoupled. For demo purposes, historical replay with a simulated clock exercises the same code path as a production cron job.
