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

- [ ] **PBI 29 — Latency SLO Baseline + CI Guard**
  - Closes the **latency arm** of the SLA: `docs/system_design.md` states `< 200ms p95 for GET /heatmap`, but nothing in the repo measures or enforces it today.
  - Add `scripts/bench_heatmap.py` that spins up the FastAPI app and hits `GET /heatmap` N times (default 500) against a local uvicorn, reporting p50 / p95 / p99 latency
  - Record the baseline in `docs/system_design.md` — replace `< 200ms target` with `< 200ms target / <X> ms measured on <commit>`
  - Add a CI smoke check (`make bench` or pytest mark) that fails when p95 exceeds a configurable threshold (default 250ms)
  - Add a `/health` field for last-recent-request p95 once PBI 27 lands (optional follow-up — keep PBI 29 itself independent)
  - Acceptance:
    - `tests/test_bench_heatmap.py` — (a) bench script runs against a fixture predictions parquet, (b) p95 stays under the configured threshold, (c) regression is detected when an artificial 300ms sleep is injected into the handler
    - `docs/system_design.md` updated with a real measured number
    - ruff + mypy clean
  - Depends on: PBI 30 (needs the GitHub Actions workflow to actually run the bench-CI smoke check; PBI 27 enriches but does not block)
  - Difficulty: S

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

## Release: Architecture Alignment with Updated Five-Lane Design (PBIs 18–27)

> **Why this exists.** The reference system design in `reference/ny_taxi_demand_prediction.html` was updated to a five-lane architecture (C00 Orchestrator, C01–C03 Ingestion with Data Validation, C04 Training with gate + registry stages + fallback, COS Inference with Feature Store, C05 Monitoring & Feedback). The README business story now also names **recall@top-K** and **WAPE-lift over a last-week-same-hour baseline** as the bridging metrics that tie ML output to driver outcome. None of those are present in the current code. PBIs 18–27 close the gap.
>
> **Why now.** Five of the eight gaps are SLA-critical (validation gate, fallback, prediction log, drift detector, schema validation). The other three (baseline+recall@K, `/health`, feature store) close the credibility gap between what the README claims and what the code does. Without PBI 18 in particular, the README's "beats the last-week-same-hour baseline" claim is unverifiable — and that's the *first* question a reviewer will ask.

### Activity 8: Business-Metrics Honesty (PBIs 18, 27)

Supports the README's "How Value Is Measured" section. Cheapest gap to close, highest credibility lift.

- [x] **PBI 18 — Baseline Model + Recall@K + WAPE-Lift Reporting** — *shipped in `783f5ff`*
  - Add `app/training/baseline.py` containing `LastWeekSameHourBaseline` — predicts each `(zone, bucket)` as the count from one week earlier in the same `(zone, bucket-of-week)` slot
  - Add `app/core/metrics.py` (also closes the duplicated-`evaluate()` portfolio finding) containing `compute_recall_at_k(predicted, actual, k=3)` and `compute_wape_lift(baseline_wape, model_wape)`
  - Wire into `Trainer.run()`: after the LGBM model is trained, also evaluate the baseline on the same test split, log both sets of metrics to MLflow under a single run with prefixes (`baseline_*`, `model_*`), and log the lift as `wape_lift_pct`
  - Acceptance:
    - `tests/test_baseline.py` — unit tests on a synthetic 4-week, 3-zone fixture: (a) baseline returns last-week's count for known rows, (b) recall@3 equals manually verified value, (c) WAPE lift positive when LGBM beats baseline
    - One MLflow run shows both `baseline_wape` and `model_wape` plus `wape_lift_pct` and `recall_at_3`
    - README gets a results table populated with real numbers from this run
  - Depends on: nothing
  - Difficulty: M

- [x] **PBI 27 — `/health` + `/model-info` + Simulated-Clock Flag** — *shipped on `refactor/app-package-layout`*
  - Add `GET /health` to `app/api/main.py` returning `{status, data_publication_lag_days, simulated, last_prediction_bucket, model_version}`
  - Add `GET /model-info` returning current model registry stage, version, training date, baseline lift
  - Add `simulated: true` to `/heatmap` response (since the 15-min cadence is replayed historical data, not live telemetry — README now states this explicitly)
  - Trainer tags promoted version with `wape_lift_pct` for `/model-info` to read
  - Acceptance: 19 acceptance tests in `tests/test_e2e_health_model_info.py` + 10 unit tests in `tests/test_serving.py`; existing `/heatmap` tests still green
  - Depends on: PBI 18 (for `baseline_lift`), PBI 20 (for `model_version` from registry)
  - Difficulty: S

### Activity 9: Data Validation Gate (PBI 19)

Closes the **Data Validation** node added to Pipeline 1 in the updated diagram. Without this, schema drift in TLC data (which has shifted columns year-over-year) silently corrupts training.

- [x] **PBI 19 — Pandera Schema Gate Between Bronze and Silver** — *shipped in `d317be0`*
  - Add `app/ingestion/schema.py` with a Pandera `DataFrameSchema` describing the expected Bronze schema (column names, dtypes, nullness, value ranges for `passenger_count`, `trip_distance`, `PULocationID`, `tpep_pickup_datetime`)
  - Wire into `DataCleaner.run()` at the start: validate the input frame; on failure, raise `DataValidationError` and halt
  - Log validation results (rows in, rows passing, rows failing per rule) to a JSON file under `data/silver/_validation/` for the future monitoring lane to consume
  - Acceptance:
    - `tests/test_clean.py::TestSchemaValidation` — (a) clean frame passes, (b) frame with missing column fails, (c) frame with out-of-range `PULocationID` fails, (d) validation log JSON is written with expected fields
    - ruff + mypy clean
  - Depends on: nothing
  - Difficulty: M

### Activity 10: Model Registry, Validation Gate, Fallback (PBIs 20, 21)

Closes the **Validation & Deployment** gating step and the **Fallback pipeline** note in C04 of the updated diagram. These are SLA-critical: today a degraded model is silently promoted and an absent model returns 503.

- [x] **PBI 20 — MLflow Model Registry + Champion-vs-Challenger Gate** — *shipped in `17960de`*
  - In `Trainer.run()`, after training and baseline evaluation (PBI 18), call `mlflow.register_model(...)` to register the model under `ny_taxi_demand_model` with a new version
  - Implement champion-vs-challenger gate: load current Production stage; if new model's test WAPE is not at least `train.gate.min_wape_lift_pct` better (default 2%), reject — do NOT transition stages; log rejection reason as MLflow tag `gate_rejected_reason`
  - If accepted: transition new version → Production; previous Production → Archived
  - Add `train.gate.min_wape_lift_pct` to `config.yml`
  - Acceptance:
    - `tests/test_train.py::TestRegistryGate` — (a) first-ever model is auto-promoted (no incumbent), (b) better challenger is promoted, (c) worse challenger is rejected and Production stays unchanged
    - MLflow registry stages observable via `mlflow models list`
  - Depends on: PBI 18 (baseline metrics needed for gate comparison)
  - Difficulty: L

- [x] **PBI 21 — Inference Fallback to Previous Approved Model** — *shipped in `ac0e73c`*
  - In `app/inference/simulator.py` and `app/api/main.py`, load model from MLflow registry Production stage; on `MlflowException` or missing-stage, fall back to most recent Archived stage and log a `model_fallback_active=true` warning
  - If both Production and Archived are empty, fail loudly (don't silently serve random data)
  - Tag predictions parquet with `model_version` and `served_from_stage` columns
  - `/heatmap` response includes the same fields per request
  - Acceptance:
    - `tests/test_inference.py::TestFallback` — (a) Production present → uses Production, (b) Production absent + Archived present → uses Archived with warning, (c) both absent → raises `NoApprovedModelError`
    - `/heatmap` response schema test asserts `model_version` + `served_from_stage` keys present
  - Depends on: PBI 20
  - Difficulty: M
  - Note: new `app/inference/exceptions.py` houses `NoApprovedModelError`; shared `tests/conftest.py::make_mock_version` extracted to deduplicate across 4 test modules; `Predictor.load_model()` rewire deferred (out of scope per story).

### Activity 11: Monitoring & Feedback Lane (PBIs 22, 23, 24)

Closes the entire C05 Monitoring & Feedback lane in the updated diagram. Without these, the validation gate has no incumbent to compare against and the retrain trigger has no signal source.

> **Sequencing note (2026-05-29).** PBI 22 (prediction log) is shipped and underpins the lane, but PBIs 23 + 24 are intentionally deferred until after PBI 30 (CI/CD) and PBI 25a (orchestrator + `Clock` abstraction). Rationale: the `Clock` abstraction in 25a unlocks the demo story and is architecturally higher-leverage than the Evidently wrapper. Monitoring will reattach via PBI 25b once 23 + 24 exist.

- [x] **PBI 22 — Prediction Log + Actuals-Join Job** — *shipped in `6d6854f`*
  - Add `app/monitoring/prediction_log.py` — appends every served `/heatmap` response to a partitioned parquet under `data/monitoring/predictions/dt=YYYY-MM-DD/`
  - Add `app/monitoring/actuals_join.py` — Airflow-friendly script that, given a date window, joins predictions with the eventual ground-truth `trip_count` from Gold and writes `data/monitoring/joined/`
  - Schema: `(bucket_time, pulocationid, predicted, actual, model_version, served_at)`
  - Acceptance: `tests/test_monitoring.py` — synthetic predictions + actuals → joined frame has expected row count and non-null `actual` where Gold data exists
  - Depends on: nothing
  - Difficulty: M
  - Shipped with 41 new tests (13 unit + 26 acceptance + 2 hook); AC 7 parametrized across `OSError`/`RuntimeError`/`ValueError`; `served_from_stage` added to the log schema alongside `model_version` so PBI 21 fallback metadata is preserved through the join.

- [ ] **PBI 23 — Drift & Performance Detector (Evidently)**
  - Add `app/monitoring/drift.py` — wraps Evidently `DataDriftPreset` (feature drift) and `RegressionPreset` (model performance) over a rolling 7-day window from PBI 22's joined frame
  - Writes a JSON report to `data/monitoring/drift/dt=YYYY-MM-DD/report.json` with one boolean per check
  - Acceptance: `tests/test_monitoring.py::TestDrift` — (a) stationary synthetic data produces no-drift report, (b) shifted distribution produces drift=True
  - Depends on: PBI 22
  - Difficulty: L

- [ ] **PBI 24 — Alerts + Retrain Trigger**
  - Add `app/monitoring/alert.py` — reads PBI 23's report; if drift OR WAPE degraded ≥ `monitoring.degradation_threshold_pct` vs the model's training-time WAPE, write a `retrain_required` marker file under `data/monitoring/triggers/`
  - The Airflow training DAG (PBI 25) watches this directory via a `FileSensor` and fires retraining on demand
  - For local development, also print a clear log line and (optionally) emit a webhook if `monitoring.webhook_url` is set
  - Acceptance: `tests/test_monitoring.py::TestAlerts` — drift report with `drift=True` → trigger file written; clean report → no file
  - Depends on: PBI 23
  - Difficulty: M

### Activity 12: Production Orchestrator (PBI 25)

Promotes the orchestrator out of `experiment/` and into a first-class lane (C00 in the updated diagram).

> **Sequencing note (2026-05-29).** Original PBI 25 has been split into **25a** (Clock + 2 DAGs) and **25b** (`monitor_hourly_dag` wiring). This lets the architecturally interesting `Clock` abstraction and the orchestrator promotion ship *before* monitoring (PBIs 23/24), per the revised plan. PBI 25b reattaches the monitoring DAG once PBI 24 exists.

- [ ] **PBI 25a — Promote Airflow DAGs to Top-Level `orchestration/` + `Clock` Abstraction (refresh + train DAGs only)**
  - **Prerequisite refactor — `Clock` abstraction (do this first, in the same PR):**
    - Add `app/core/clock.py` defining a `Clock` protocol with `now() -> datetime` and `sleep(seconds: float) -> None`
    - Two implementations: `WallClock` (production — wraps `datetime.utcnow()` and `time.sleep`) and `SimulatedClock(start: datetime, speedup: float = 450.0)` (demo — advances 15 simulated minutes every 2s of wall-clock by default)
    - Every component that currently calls `datetime.utcnow()` / `datetime.now()` / `time.sleep` — ingestion watermark, Airflow sensor marker logic, predictor's "current bucket" lookup, API's "latest hour" fallback — is refactored to receive a `Clock` and read time through it
    - One repo-wide grep regression test in `tests/test_clock_discipline.py` asserts that no production module under `app/` imports `datetime.utcnow`, `datetime.now`, or `time.sleep` directly (allowlist: `app/core/clock.py` only)
    - Selection: `clock = SimulatedClock(...)` when env var `DEMO_MODE=1`, else `WallClock()` — picked up at app startup
  - Move `experiment/orchestration/` → `orchestration/` at repo root
  - Add **two** production DAGs (both driven by the `Clock`, not `datetime.utcnow()`):
    1. `refresh_15min_dag.py` — ingest → validate (PBI 19) → feature → predict → log (PBI 22); idempotent on `(date, bucket)` watermark
    2. `train_monthly_dag.py` — cron `@monthly` + manual; train → gate (PBI 20) → register
  - **Deferred to PBI 25b:** `monitor_hourly_dag.py` (depends on PBIs 23 + 24 existing)
  - Add `orchestration/README.md` with DAG topology, triggering semantics, SLAs, and a "Demo vs Production" section explaining the clock selection. Document the deferred monitoring DAG with a TODO referencing PBI 25b so reviewers see the lane is planned, not abandoned.
  - Acceptance:
    - `airflow dags list` shows the two DAGs after `make airflow-init`
    - DAGs parse cleanly under `pytest tests/test_dag_integrity.py` (existing pattern)
    - Idempotency: rerunning the same bucket-date pair is a no-op
    - Clock-discipline test green: no stray `utcnow()` / `time.sleep` outside `app/core/clock.py`
    - `DEMO_MODE=1 make demo` brings up the full stack on `SimulatedClock` and the `/heatmap` payload's `bucket_time` advances by 15 simulated minutes per ~2s of wall-clock
  - Depends on: PBI 20 (registry stage label for `train_monthly_dag`'s gate step), PBI 30 (CI gates protect the refactor)
  - Difficulty: L

- [ ] **PBI 25b — Wire `monitor_hourly_dag` Into Orchestrator**
  - Add `orchestration/dags/monitor_hourly_dag.py` — runs PBI 23 drift detector; PBI 24 alert step; `FileSensor` on `data/monitoring/triggers/` fires `train_monthly_dag` on retrain demand
  - Update `orchestration/README.md` to remove the TODO from PBI 25a and document the monitor lane as live
  - Acceptance:
    - `airflow dags list` now shows three DAGs
    - `tests/test_dag_integrity.py` covers the new DAG
    - End-to-end: hand-craft a drift report with `drift=True` under `data/monitoring/drift/`; trigger `monitor_hourly_dag`; assert the `train_monthly_dag` is queued via the `FileSensor` hand-off
  - Depends on: PBI 24, PBI 25a
  - Difficulty: S

### Activity 13: Feature Store (PBI 26) — *stretch*

Closes the **Feature Store** node added to the inference lane. Only needed if true on-demand inference becomes a requirement; current batch-then-serve is acceptable for the portfolio scope.

- [ ] **PBI 26 — In-Memory Online Feature Store**
  - Add `app/inference/feature_store.py` — a simple in-memory cache keyed by `(pulocationid, bucket_time)` returning the lag features needed for prediction
  - The refresh DAG (PBI 25) writes the latest features to this store after each ingestion run
  - Inference Service (`/heatmap`) reads from the store instead of (or in addition to) the precomputed predictions parquet, enabling arbitrary `?bucket_time=` queries
  - Acceptance: integration test that adds a feature, queries it, asserts round-trip; latency stays sub-200ms
  - Depends on: PBI 25
  - Difficulty: L
  - Note: defer until product asks for on-demand inference; otherwise YAGNI

### Activity 14: Driver-Agent Simulation Harness (PBI 28)

> **Why this exists.** The README claims the system "guides drivers to higher-demand zones" and that following the heatmap beats following a last-week-same-hour intuition. Today that claim is unverified — it follows from model-quality metrics (MAE/WAPE) but never from a *driver-outcome* measurement. PBI 28 closes that gap: a closed-loop synthetic-driver simulation that replays a held-out week of historical TLC data, gives N synthetic drivers different policies, and reports per-policy trips served, idle time, and miles-to-pickup. This is what turns "the model is accurate" into "drivers following the model are better off."
>
> **Why it's separate from PBI 25.** PBI 25 is *system-level* simulation (the pipeline runs on a simulated clock). PBI 28 is *business-level* simulation (synthetic drivers act on the system's output and we measure outcomes). They are independent and complementary.

- [ ] **PBI 28 — Driver-Agent Simulation Harness**
  - Add `app/simulation/driver_agents.py` with:
    - `Driver` dataclass: `{driver_id, lat, lon, state: Literal["idle", "en_route", "on_trip"], current_zone, destination_zone, trip_ends_at}`
    - `Policy` protocol: `choose_next_zone(driver, heatmap_response, clock) -> zone_id`
    - Three policies:
      1. `HeatmapFollower` — calls `/heatmap`, picks one of the top-K zones (configurable K) weighted by predicted count
      2. `LastWeekFollower` — baseline; reads PBI 18's last-week-same-hour lookup, picks top-K zones from *that*
      3. `Stationary` — control; never moves
  - Add `app/simulation/harness.py`:
    - Spawns N drivers (default 50), distributes them across NYC zones with realistic starting positions (sampled from historical pickup density)
    - At each simulated 15-min tick (driven by the same `SimulatedClock` from PBI 25):
      1. For each `idle` driver: invoke its policy, set destination, transition → `en_route`, compute ETA from haversine distance / average NYC taxi speed (use 12 mph default)
      2. For each `en_route` driver whose ETA has passed: transition → `idle` at the destination zone
      3. For each `idle` driver: check the historical TLC trip table for trips that *actually started* in that driver's zone during this 15-min window; if any are unclaimed by other drivers in that zone, assign one (FIFO by `driver_id`); transition → `on_trip` with `trip_ends_at = pickup_time + historical_trip_duration`
      4. For each `on_trip` driver whose `trip_ends_at` has passed: transition → `idle` at the historical dropoff zone
    - Records per-driver, per-tick metrics to `data/simulation/runs/<run_id>/events.parquet`
  - Add `app/simulation/report.py`:
    - Aggregates per-policy: total trips completed, % idle time, average miles-to-pickup, total earnings (use `fare_amount + tip_amount` from historical data on the assigned trip)
    - Emits a Markdown results table to `data/simulation/runs/<run_id>/report.md` and a CSV under the same dir
    - The CI workflow (PBI 25 era) runs this harness on a tiny synthetic week as a regression test; the full week-long run is documented in the README
  - Acceptance:
    - `tests/test_driver_simulation.py`:
      1. `test_heatmap_follower_visits_top_k` — synthetic 5-zone heatmap with zone 3 at top → idle driver's `destination_zone` is in {top-K}
      2. `test_trip_assignment_is_exclusive` — two drivers in the same zone, one historical trip → exactly one driver goes `on_trip`, the other stays `idle`
      3. `test_stationary_policy_serves_only_local_trips` — Stationary driver in a low-demand zone serves fewer trips than HeatmapFollower in the same starting position over the same week
      4. `test_report_aggregation` — synthetic events frame → report row counts and totals match hand-computed numbers
    - One full run on a held-out week of TLC data lands the per-policy table in the README under "Results" — with non-zero lift for `HeatmapFollower` over `LastWeekFollower` over `Stationary`, or an honest write-up explaining why not
    - `data/simulation/runs/` and `data/simulation/` added to `.gitignore`; the latest `report.md` is committed under `docs/simulation_results.md` as the canonical artifact
  - Depends on: PBI 18 (needs `LastWeekSameHourBaseline` for the comparison policy), PBI 25a (needs `SimulatedClock` so the harness shares the same time abstraction as the live pipeline)
  - Difficulty: L
  - **Honest limitation to document in the harness README:** the simulation is *single-agent-equivalent* — all N drivers see the same heatmap and historical trips are a fixed pool, so this captures recall-style lift but not demand-supply equilibration. A future PBI could add a supply-aware demand model.

### Activity 15: GitHub Actions CI/CD (PBI 30)

> **Why this exists.** Today the project documents `uv run ruff check`, `uv run ruff format --check`, `uv run mypy`, and `uv run pytest` as the manual gates (README:186–196), but nothing enforces them on PRs. The only existing GitHub Actions workflow is `docs.yml` (mkdocs deploy). PRs into `main` can land lint/type/test regressions silently. This PBI wires the manual gates into automated CI so the rules in `.claude/rules/coding-style.md` and `.claude/rules/testing.md` are mechanically enforced.
>
> **Why now.** Phase A is shipped (PBIs 18, 19, 20, 27 all live on branches awaiting merge). Future PBIs (21, 22–24, 25, 28) all add code paths that need the same gates. Adding CI before those land prevents the maintenance debt of fixing regressions after the fact. Also gates PBI 29's latency-SLO test (`bench_heatmap_p95` cannot be a useful guardrail without a workflow to run it on every PR).

- [x] **PBI 30 — GitHub Actions CI/CD: Lint + Test (+ Docker, deferred)** — *shipped in `98761fd`* (PR #13 → dev_MVP; 5 files, 20 acceptance tests; pre-existing `test.ipynb` ruff-format failure and 7 tests/ mypy errors deferred to follow-up PBIs)
  - Add `.github/workflows/lint.yml`:
    - Triggers: `pull_request` against any branch; `push` to `main`
    - Steps: checkout → `astral-sh/setup-uv@v3` (with cache) → `uv python install 3.13` → `uv sync --dev` → `uv run ruff check .` → `uv run ruff format --check .` → `uv run mypy app tests`
    - Must pass without modification on the current `main` after PBIs 20/27 merge
  - Add `.github/workflows/test.yml`:
    - Triggers: `pull_request` against any branch; `push` to `main`
    - Steps: checkout → setup-uv → `uv sync --dev` → `uv run pytest tests/ -v --tb=short`
    - **Torch handling**: verify via `grep -r "import torch" app/ tests/` whether torch is actually used by the test suite. If not, add `--no-install-package torch --no-install-package torchvision` to `uv sync` to save ~2 GB and ~90s of install on CPU runners. If torch IS imported, add a CPU-only override (`UV_INDEX_STRATEGY=unsafe-first-match` + `--extra-index-url https://download.pytorch.org/whl/cpu`)
    - **Pre-existing failure**: `tests/test_dataset.py::test_load_data` fails with `KeyError: 'raw_path'` (documented in MEMORY.md). Either fix the fixture or mark `@pytest.mark.xfail(reason="known: requires raw_path config key")` before turning CI red
  - Configure branch protection on `main` (repo settings, not a workflow file):
    - Require `lint` and `test` workflows to pass before merge
    - Require at least 1 PR review
    - Disallow direct pushes to `main` (matches `.claude/rules/git-workflow.md`)
  - **Deferred — Docker pipeline**: do NOT add `.github/workflows/docker.yml` in this PBI. There is no `Dockerfile` for the FastAPI service today (only `experiment/orchestration/docker-compose.yml` for the Airflow demo). When a Dockerfile is written, follow up with a `docker.yml` that builds and pushes to `ghcr.io/<owner>/<repo>` on tag pushes (`on: push: tags: ['v*']`) using `docker/build-push-action@v5` with GHA cache. Capture this as a separate PBI when a deploy target is chosen.
  - Acceptance:
    - Open a draft PR against `main`; both `lint` and `test` checks appear and pass green on the current tree (after handling `test_load_data`)
    - Open a second PR that introduces a deliberate ruff violation (e.g. unused import); `lint` workflow fails with the rule code in the log
    - Open a third PR that breaks a serving test; `test` workflow fails with the failing test name in the log
    - Branch protection prevents merging any of the three PRs while a required check is red
    - Cold-cache CI run completes in under 4 minutes (lint) + under 5 minutes (test); warm-cache under 1 minute each
    - README gets a CI badge row pointing at the two workflows
  - Depends on: nothing (independent of feature PBIs)
  - Difficulty: S
  - **Out of scope for this PBI:** Docker build/publish workflow, coverage upload (Codecov/Coveralls), pre-commit hooks, the latency-SLO benchmark workflow (that lives in PBI 29), MLflow tracking-server-based registry tests in CI

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

Architecture-Alignment (PBIs 18–28 — closes gaps vs updated five-lane diagram + driver outcome):
PBI 18 (baseline + recall@K) ──┬──> PBI 20 (registry + gate) ──> PBI 21 (fallback) ──┐
                                │                                                       │
                                ├──> PBI 27 (/health + /model-info)                    │
                                │                                                       │
                                └────────────────────────────────────────────────────┐ │
                                                                                      │ │
PBI 19 (Pandera schema gate)  ───────────────────────────────────────────────────────┼─┤
                                                                                      │ │
PBI 22 (prediction log + actuals) ──> PBI 23 (drift detector) ──> PBI 24 (alerts) ──┐│ │
                                                                                      ││ │
                                                                                      ▼▼ ▼
                                                              PBI 25 (production orchestrator
                                                                       + SimulatedClock)
                                                                                        │
                                                                ┌───────────────────────┼──> PBI 26 (feature store, stretch)
                                                                │                       │
                                                                └──> PBI 28 (driver-agent harness — needs PBI 18 baseline policy
                                                                              + PBI 25 SimulatedClock)
```

## Priority Order

> `[x]` = shipped (struck through); `[ ]` = open. Ordered by execution sequence: shipped first (in the order they landed), then open work grouped by phase.

### Shipped — MVP & B-track (forecasting path)

| Done | Order | PBI | Rationale |
|------|-------|-----|-----------|
| [x] | 1 | ~~PBI 1~~ | ~~Validate existing single-horizon model end-to-end~~ |
| [x] | 2 | ~~PBI 2~~ | ~~Inference pipeline — completes the full ML loop~~ |
| [x] | 3 | ~~PBI 3~~ | ~~Heatmap API — serves predictions to drivers (MVP complete)~~ |
| [x] | 4 | ~~B0~~ | ~~Synthetic E2E test suite — pins legacy, RED-locks B1~~ |
| [x] | 5 | ~~B1~~ | ~~30-min nowcast + neighbor-mean smoothing — superseded by B2 as default; method preserved~~ |
| [x] | 6 | ~~B2~~ | ~~15-min refresh + 30-min summed-horizon target — current MVP forecasting path~~ |

### Shipped — Performance Hardening

| Done | Order | PBI | Rationale |
|------|-------|-----|-----------|
| [x] | 7 | ~~PBI 12~~ | ~~Bronze temporal-sort invariant test — TDD RED step; safe in isolation~~ |
| [x] | 8 | ~~PBI 13~~ | ~~Parallel-equivalence regression test — RED for PBI 14~~ |
| [x] | 9 | ~~PBI 14~~ | ~~Parallel `load_raw` — closes the "parallelize I/O-bound" gap~~ |

### Open — Phase A · Credibility (start here, ~10h)

| Done | Order | PBI | Rationale |
|------|-------|-----|-----------|
| [x] | 10 | ~~PBI 18~~ | ~~Baseline + recall@K + WAPE-lift — turns the README's value claim into a reproducible number; unblocks gate (PBI 20)~~ — *shipped in `783f5ff`* |
| [x] | 11 | ~~PBI 19~~ | ~~Pandera schema gate — closes the Data Validation node; independent; ship in parallel with PBI 18~~ — *shipped in `d317be0`* |
| [x] | 12 | ~~PBI 27~~ | ~~`/health` + `/model-info` + simulated flag — small but visible; depends on PBI 18 + PBI 20~~ — *shipped in `ba14223` (follow-ups) on top of `adc4ad7`* |

### Open — Phase B · SLA-critical safety (~14h)

| Done | Order | PBI | Rationale |
|------|-------|-----|-----------|
| [x] | 13 | ~~PBI 20~~ | ~~MLflow registry stages + validation gate — first SLA-critical fix; depends on PBI 18 for the comparison metric~~ — *shipped in `17960de`* |
| [x] | 14 | ~~PBI 21~~ | ~~Inference fallback — completes C04 fallback note; depends on PBI 20~~ — *shipped in `ac0e73c`* |

### Open — Phase C · CI hardening (do first — protects everything that follows, ~6h)

| Done | Order | PBI | Rationale |
|------|-------|-----|-----------|
| [x] | 15 | ~~PBI 30~~ | ~~GitHub Actions CI/CD (lint + test) — independent, S-sized; every PBI from now on lands through enforced gates. **Resequenced 2026-05-29 to land before PBI 25a so the orchestrator refactor ships under CI.**~~ — *shipped in `98761fd`* |

### Open — Phase D · Production orchestrator + `Clock` (~16h)

| Done | Order | PBI | Rationale |
|------|-------|-----|-----------|
| [ ] | 16 | PBI 25a | Promote Airflow DAGs to `orchestration/` + `Clock` abstraction (refresh + train DAGs only). **Resequenced 2026-05-29 ahead of PBIs 23/24** — the `Clock` abstraction is architecturally higher-leverage than the Evidently wrapper; monitoring DAG reattaches in PBI 25b. Depends on PBI 20, PBI 30. |

### Open — Phase E · Closed-loop monitoring (~24h)

| Done | Order | PBI | Rationale |
|------|-------|-----|-----------|
| [x] | 17 | ~~PBI 22~~ | ~~Prediction log + actuals join — foundation for monitoring lane (C05)~~ — *shipped in `6d6854f`* |
| [ ] | 18 | PBI 23 | Drift detector (Evidently) — depends on PBI 22 |
| [ ] | 19 | PBI 24 | Alerts + retrain trigger — closes the monitoring → orchestrator feedback arrow; depends on PBI 23 |
| [ ] | 20 | PBI 25b | Wire `monitor_hourly_dag` into the orchestrator now that PBI 24 exists; depends on PBI 24 + PBI 25a |

### Open — Phase F · Closed-loop simulation

| Done | Order | PBI | Rationale |
|------|-------|-----|-----------|
| [ ] | 21 | PBI 28 | Driver-agent simulation harness — proves the "drivers following the heatmap are better off" claim; depends on PBI 18 + PBI 25a |

### Open — API & ingestion hardening

| Done | Order | PBI | Rationale |
|------|-------|-----|-----------|
| [ ] | 22 | PBI 16 | Paginate `/heatmap` — closes the "pagination for large responses" gap |
| [ ] | 23 | PBI 29 | Latency SLO baseline + CI guard — closes the latency arm of the SLA; depends on PBI 30 for the CI workflow to run the benchmark |
| [ ] | 24 | PBI 15 | Streaming bronze write — defer until ingestion OOMs in practice |
| [ ] | 25 | PBI 17 | Socrata HTTP compliance — blocked until downloader migrates into `app/` |

### Open — Post-MVP (personalized ranking + reliability)

| Done | Order | PBI | Rationale |
|------|-------|-----|-----------|
| [ ] | 26 | PBI 5 | Zone lookup — needed by personalized ranking |
| [ ] | 27 | PBI 6 | Location capture — simple, enables testing |
| [ ] | 28 | PBI 7 | Scoring function — pure math, unit testable |
| [ ] | 29 | PBI 8 | Recommendation API — personalized ranking endpoint |
| [ ] | 30 | PBI 11 | Simulated streaming — shows production awareness (now largely subsumed by PBI 25a's `SimulatedClock`) |
| [ ] | 31 | PBI 9 | Stale-data handler — historical-averages fallback; complementary to PBI 21's model-level fallback |
| [ ] | 32 | PBI 10 | Monitoring (L5 signal) — superseded by PBIs 22–24 which decompose this into prediction log + drift + alerts |
| [ ] | 33 | PBI 4 | Multi-horizon — deferred; A3 knob (`horizon_buckets`) already shipped, just needs a training config flip |

### Stretch

| Done | Order | PBI | Rationale |
|------|-------|-----|-----------|
| [ ] | 34 | PBI 26 | Online feature store — defer until on-demand inference is actually needed |

## Recommended Sequencing for PBIs 18–30

> **Revised 2026-05-29.** The original plan ran monitoring (23/24) before orchestration (25). After landing PBI 22 it became clear that the `Clock` abstraction inside PBI 25 is architecturally higher-leverage than the Evidently wrapper, and that landing CI (PBI 30) first protects every subsequent refactor. PBI 25 was therefore split into 25a (Clock + 2 DAGs) and 25b (`monitor_hourly_dag` wiring after 24 exists).

**Phase A — Credibility (highest leverage, ~10h total):**
1. **PBI 18** (baseline + recall@K) — proves the model actually adds value over a naive baseline. Without this, every other PBI is built on an unproven foundation. *Shipped.*
2. **PBI 19** (Pandera schema gate) — ship in parallel; independent. *Shipped.*
3. **PBI 27** (`/health` + `/model-info`) — visible API surface, low effort. *Shipped.*

**Phase B — SLA-critical safety (~14h total):**
4. **PBI 20** (registry + gate) — *Shipped.*
5. **PBI 21** (inference fallback) — *Shipped.*

**Phase C — Prediction log foundation (~10h):**
6. **PBI 22** (prediction log + actuals join) — *Shipped.* Foundation for monitoring; sits idle until PBIs 23/24 are revisited in Phase E.

**Phase D — CI hardening (do this NEXT, ~6h):**
7. **PBI 30** (GitHub Actions lint + test) — independent, S-sized. Every PBI from here on lands under enforced gates. Doing this *before* PBI 25a means the orchestrator refactor ships with CI protection.

**Phase E — Production orchestrator + `Clock` (~16h):**
8. **PBI 25a** (orchestrator promotion + `Clock` abstraction; refresh + train DAGs only) — *system-level* simulation layer that lets a reviewer watch the heatmap update on a compressed clock. `monitor_hourly_dag` deliberately deferred to 25b.

**Phase F — Closed-loop monitoring (~24h):**
9. **PBI 23** (drift detector — Evidently `DataDriftPreset` + `RegressionPreset` over PBI 22's joined frame)
10. **PBI 24** (alerts + retrain trigger — emits the `retrain_required` marker file)
11. **PBI 25b** (wire `monitor_hourly_dag` into the orchestrator — closes the C05 lane)

**Phase G — Closed-loop simulation:**
12. **PBI 28** (driver-agent harness — *business-level* simulation that produces the per-policy trips-served table the README references)

**Phase H — Stretch:**
13. **PBI 26** (online feature store — only if on-demand inference becomes a requirement)

## Known Limitations & Future Considerations

- **Feedback loop:** Driver recommendations influence future trip data (see DiDi's multi-driver optimization). Not addressed in MVP since heatmap approach + surge pricing mitigate herding naturally.
- **Multi-driver coordination:** MVP serves the same heatmap to all drivers (Uber model). DiDi-style per-driver optimization requires fleet state tracking and is deferred to post-MVP.
- **A/B testing:** No framework for measuring if recommendations improve driver outcomes. Design doc recommended before scaling.
- **Batch vs streaming inference:** Prediction is batch by nature — the model needs aggregates over completed time windows (e.g., hourly trip counts). Data ingestion can be streaming into a DB, but inference runs as a periodic batch job (every 15 min in Uber's case). These two loops are decoupled. For demo purposes, historical replay with a simulated clock exercises the same code path as a production cron job.
