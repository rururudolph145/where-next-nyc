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

- [ ] **PBI 4 — Multi-Horizon Forecasting (t+30, t+60)** — *MVP-deferred; superseded by B1*
  - Original plan: extend training to produce shifted targets (trip_count at t+30, t+60)
  - Status: replaced for MVP by the simpler **B1 nowcast** path (see Activity 3b). Multi-horizon shift remains available via the already-shipped `add_shifted_target` knob (PBI A3, commit `4ea0a03`) if/when product asks for true forecasting.
  - Input: Gold layer features with shifted targets
  - Output: predicted trip_count per zone per horizon
  - Acceptance: separate models or multi-output model; metrics logged per horizon

### Activity 3b: Nowcast MVP (30-min bucket + spatial smoothing) — *B-track*

> **Why this exists.** End-to-end smoke testing of the 15-min + shifted-horizon design (A1-A3 shipped: commits `6ab31f1`, `e546e3d`, `4ea0a03`) surfaced that bucketed counts are sparse at fine granularity (busiest zones: 6-15 trips/30-min bucket; most zones: 0-3). Coefficient of variation σ/μ = 1/√λ doubles when buckets are quartered. Pranshu's notebook (`experiment/NYC_Yellow_Taxi_Pickup_Density.ipynb`) achieves ~13% MAPE at 10-min binning *only* because it coarsens space to 40 k-means clusters. We retain the 263 TLC zones, so we coarsen time instead (30-min) and add spatial smoothing.
>
> **Framing.** Predict `trip_count` for the current 30-min bucket `[t, t+30min)` from features observable at `t` — no target shift required, because the lag features (`lag_1`, `lag_2`, …) reference *prior* buckets and the trainer's explicit feature list excludes the target. L2 loss on the count. The driver-facing "lead time" is naturally 0-30 min (whatever fraction of the bucket has elapsed when the prediction is served).
>
> **Supersedes** A1-A5 plan for MVP: A1 (`bucket` knob) is reused; A2 (lag generalization) and A3 (shifted target) are kept in the codebase but `horizon_buckets` stays 0; A4 (Poisson loss) and A5 (spatial-spillover) are subsumed by B1's design choices (L2 + k-NN smoothing).

- [x] **B1 — 30-Min Bucket + Neighbor-Mean Lag Smoothing** — *shipped*
  - Config changes (`config.yml`):
    - `feature.bucket: "30min"` (was `"h"`)
    - `feature.lags: [1, 2, 4, 6, 12, 48, 336]` (1×30min, 1h, 2h, 3h, 6h, 1d, 1w) — replaces hourly lag schedule
    - `feature.horizon_buckets: 0` (unchanged — pure nowcast)
    - `feature.n_neighbors: 5` (new — k for spatial smoothing)
    - `train.features`: add `trip_count_lag_1_neighbor_mean`; lag column names remain `trip_count_lag_{n}` (now 30-min offsets)
  - Code changes (`app/ingestion/feature.py`):
    - New method `FeatureEngineer.add_neighbor_lag_smoothing(df, k=5)`:
      - Builds `sklearn.neighbors.BallTree` over zone centroids (lat, lon in radians) with `metric="haversine"`
      - For each zone, queries k+1 neighbors (self + k); drops self
      - Per `(zone, hour)` row, computes `mean(trip_count_lag_1 of k nearest neighbors at same hour)` → column `trip_count_lag_1_neighbor_mean`
      - Mean (not sum) so the feature stays on the same count scale as `trip_count_lag_1`
    - Wire into `run()` *after* `add_lag_features` and *after* `add_zone_coordinates` (needs lat/lon present, lags present, post-shapefile-orphan drop for stable neighbor sets)
  - Test plan (`tests/test_feature.py` — new `TestNeighborLagSmoothing` class):
    1. `test_neighbor_mean_matches_manual_calculation` — synthetic 6-zone grid on a line, k=2, hand-verify the neighbor-mean
    2. `test_neighbor_mean_scale_preserved` — `trip_count_lag_1_neighbor_mean.mean()` ≈ `trip_count_lag_1.mean()` (within 5%) on a uniform-zone fixture; locks the "mean not sum" choice
    3. `test_neighbor_mean_no_self_inclusion` — synthetic zone with lag_1=100 surrounded by zeros: its neighbor-mean must be 0, not include itself
    4. `test_orphan_zones_dropped` — zones missing from shapefile (e.g. 57, 264, 265) absent from output, no NaN neighbor-means
  - Acceptance:
    - All 4 unit tests + 3 synthetic E2E tests (B0 below) green
    - On the 3.8M-row 7-day Silver slice: Gold output has ≥ 1 row per (zone, 30-min bucket), no NaN in `trip_count_lag_1_neighbor_mean`, neighbor-mean column shape matches trip_count_lag_1
    - Trainer runs to completion against the new Gold schema; val/test MAE ≤ current hourly baseline + 20% tolerance (sparser bucket is harder; smoothing should compensate)
    - ruff + mypy clean on `app/ tests/`

- [x] **B0 — Synthetic-Data E2E Test Suite** (test-first, landed before B1) — *shipped*
  - New file `tests/test_e2e_synthetic.py`:
    1. `test_legacy_pipeline_smoke` — bucket=h, horizon=0, no neighbor feature: full Silver→Gold→Train→Predict on a synthetic ~50-row Silver fixture; asserts non-empty predictions, expected schema. **GREEN today** — pins legacy behaviour before B1 touches it.
    2. `test_b1_pipeline_smoke` — bucket=30min, horizon=0, n_neighbors=5: same pipeline; asserts `trip_count_lag_1_neighbor_mean` column present and finite. **RED until B1 implementation lands** (`@pytest.mark.xfail(strict=True)`).
    3. `test_no_target_leakage` — invariant check across both modes: no feature column equals `trip_count` of the same row; lag columns reference strictly prior buckets; neighbor-mean reads `lag_1`, never `trip_count`. Static schema assertion + a generated-data ablation (corrupt one lag, assert model error rises).
  - Acceptance: B0 tests written and committed; (1) green, (2) xfail-RED, (3) green on current code. Pytest count grows by 3.

- [x] **B2 — 15-Min Bucket + 30-Min Summed-Horizon Target** — *shipped*
  - **Motivation.** B1 served as a 30-min nowcast (target = current 30-min bucket). Product framing for the driver UX is "what will demand be over the next 30 minutes from *right now*?" — i.e. refresh every 15 min and predict the *sum* of the next two 15-min bins. This is still a nowcast (target window starts at `t`, the prediction time) but extends 30 min into the future — an "extended nowcast" in the taxonomy below.
  - **Nowcast vs forecast.** B1 target = `[t, t+30)` (one 30-min bin) — pure nowcast, window fully concurrent with prediction time. B2 target = `[t, t+30)` as a *sum of two 15-min bins* — extended nowcast: the second bin spans `[t+15, t+30)` which is in the future relative to the most recent observed lag bin. True forecast = `[t+30, t+60)` (zero overlap with `t`) — explicitly out of scope for MVP.
  - Config changes (`config.yml`):
    - `feature.bucket: "30min"` → `"15min"`
    - `feature.lags: [1, 2, 4, 6, 12, 48, 336]` → `[1, 2, 4, 8, 12, 24, 96, 672]` (recomputed in 15-min units: 15min, 30min, 1h, 2h, 3h, 6h, 1d, 1w)
    - `feature.horizon_sum_buckets: 2` (new — number of future buckets to sum into the target)
    - `feature.n_neighbors: 5` (unchanged — B1 spatial smoothing preserved)
    - `train.target: "trip_count"` → `"target_sum_t1_t2"`
    - `train.features`: lag column names unchanged (now 15-min offsets); `trip_count_lag_1_neighbor_mean` preserved
  - Code changes (`app/ingestion/feature.py`):
    - New init knob `self.horizon_sum_buckets` (default 0 — disables, so legacy configs untouched)
    - New method `add_summed_horizon_target(df, zone_col, target_col)`:
      - For `k in 1..N`: build `df.groupby(zone_col)[target_col].shift(-k)` (negative = forward in time)
      - Sum across the N shifted columns into `target_sum_t1_t{N}`
      - Drop the trailing N rows per zone where any contributing shift is NaN
    - Wire into `run()` *after* `add_shifted_target` (A3) and *before* `add_zone_coordinates` — independent of A3, both columns emitted when both knobs set
  - Test plan (`tests/test_feature.py` — new `TestSummedHorizonTarget` class):
    1. `test_sum_of_two_future_buckets` — synthetic 6-row zone with known counts, hand-verify each row's `target_sum_t1_t2`
    2. `test_shift_direction_is_forward` — corrupt a future bucket, assert it changes the current row's target; anti-leakage guard
    3. `test_no_cross_zone_bleed` — two zones interleaved; assert zone A's target sums only zone A's future buckets
    4. `test_horizon_sum_zero_is_noop` — `horizon_sum_buckets=0` leaves the frame untouched (B1 default path)
    5. `test_trailing_rows_dropped` — output is shorter than input by exactly N rows per zone
  - E2E test (`tests/test_e2e_synthetic.py`): `test_b2_pipeline_smoke` — full Silver→Gold→Train against B2 config, asserts `target_sum_t1_t2` column present and trainer converges.
  - Acceptance:
    - All 5 unit tests + 1 E2E test + B1 + B0 tests green (44 total in `test_feature.py` + `test_e2e_synthetic.py`)
    - ruff + mypy clean on `app/ tests/`
    - `config.yml` flipped to B2 defaults so `python -m pipeline.run_all` produces the B2 Gold schema

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

- [x] **PBI 14 — Parallelize `load_raw` with `ThreadPoolExecutor`** — *shipped in a follow-up commit*
  - Replace the sequential `for pq_file in parquets` loop in `app/ingestion/ingest.py` with `ThreadPoolExecutor.map(_read_one, paths)` (inner helper applies the rename map so parallelization is column-safe)
  - Add `ingest.max_workers: 8` to `config.yml`; threaded through `Ingestion.__init__` via `.get(..., 8)` so legacy configs without the key still work
  - After `pd.concat`, call `df.sort_values("tpep_pickup_datetime", kind="mergesort").reset_index(drop=True)` to restore canonical time order (guarded by `if "tpep_pickup_datetime" in df.columns` for schema-flexibility)
  - Acceptance: PBIs 12 + 13 pass; pytest 58 → 59 passing; ruff + mypy clean on `app/ tests/`
  - Informal benchmark (30 daily files × 50k rows = 1.5M rows): max_workers=1 → 482ms baseline, max_workers=4 → 186ms (**2.6× speedup**), max_workers=8 → 201ms (plateau). Default of 8 keeps headroom for larger production files.
  - Side fix: PBI 13's reference-dtype alignment cast to `[us]` was wrong (pre-existing `_enforce_schema` produces `[ns]` via `pd.to_datetime`); corrected to use `pd.to_datetime(...)` for both columns. Hidden by xfail-strict before PBI 14 landed.

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
                                  ├──> PBI 4 (multi-horizon) — superseded for MVP by B-track
                                  │
                                  │   A1 (bucket knob, shipped) ──┐
                                  │   A2 (lag units, shipped) ────┼──> B0 (E2E synthetic tests) ──> B1 (30-min + neighbor smoothing) ──> B2 (15-min + 30-min summed target)
                                  │   A3 (shift target, shipped) ─┘
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
| 4a | B0 | Synthetic E2E test suite — pins legacy, RED-locks B1 |
| 4b | B1 | 30-min nowcast + neighbor-mean smoothing — superseded by B2 as default; method preserved |
| 4c | B2 | 15-min refresh + 30-min summed-horizon target — current MVP forecasting path |
| 4d | PBI 4 | Multi-horizon — deferred; A3 knob (`horizon_buckets`) already shipped |
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
