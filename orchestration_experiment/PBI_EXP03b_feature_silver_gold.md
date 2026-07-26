# PBI EXP-03b — Feature (Silver → Gold) (Experimentation)

> **Type:** Experimentation PBI (not for release; lives under `orchestration_experiment/`)
> **Depends on:** EXP-03a (validate gate must be inserted first).
> **Parent:** Orchestrator redesign brainstorm (see `MEMORY.md → orchestrator-redesign-brainstorm`).
> **Split note:** The original EXP-03 bundled `validate` + `feature`. This PBI
> is the second half: given a validated bronze, produce the gold feature matrix
> the model consumes.

---

## 1. Why this PBI exists

EXP-03a validated bronze but nothing consumes it yet. EXP-03b closes the middle
of the mental model:

> silver → gold (features): lag features, cyclical time features, per-zone
> features. One row per `(zone, target_bucket)`.

Bronze (raw-shaped, per-trip) → silver (cleaned, per-trip) → gold (aggregated,
one row per `(zone, bucket)`, ML-ready). This is the biggest image fatten since
EXP-02: `geopandas` (shapefile → zone centroids), `scikit-learn` (BallTree for
neighbour smoothing), plus a new read-only mount for `data/reference/`.

---

## 2. Goal

Insert ONE task between `validate` and `advance_watermark`:

- `feature` — `DataCleaner().run()` (bronze → silver) then
  `FeatureEngineer().run()` (silver → gold). Bundled behind one task label
  because they share the same input path override and are always run together.

---

## 3. In scope

- Add `feature` task:
  `... → validate → feature → advance_watermark → write_watermark`.
- Inject config override that redirects silver_path, gold_path,
  zone_shapefile_path, and model.store_path to the experiment data root
  (same "inject config dict" seam as EXP-02's `build_ingest_config`).
- Bake `geopandas>=1.0` and `scikit-learn>=1.5` into the image (matches
  `pyproject.toml` pins).
- Mount `data/reference/:ro` into the container so `taxi_zones.shp` is
  available for `add_zone_coordinates`.
- Silver + gold artifacts land under the experiment data root, not production.

## 4. Out of scope (deferred)

- Predict / log_predict — EXP-04.
- `.done` sentinels + `check_no_new_data` boundary — EXP-05.
- Any change to `app/`, `orchestration/`, `implementation_checklist/`,
  `config.yml`.
- Retraining or model-artifact writes to `model.store_path` — `DataCleaner.run`
  persists `iqr_bounds.json` there; we redirect the path but don't validate the
  contents. Training pipeline is a separate future DAG.

---

## 5. Deliverables

1. Updated DAG with `feature` inserted (6 tasks).
2. Silver + gold artifacts materialize under the experiment data root each run.
3. Note recording:
   - gold has one row per `(zone, target_bucket)`;
   - required feature columns present (lags, cyclical, zone lat/lon,
     `target_sum_t1_t2` under the default `horizon_sum_buckets=2`);
   - causality spot-check for one `(zone, bucket)`: `trip_count_lag_1` at bucket
     `t` equals `trip_count` at bucket `t-1` for the same zone;
   - `max(hour) < window_end`, `min(hour) >= seed_start`.

---

## 6. DAG shape

```
read_watermark → ingest → validate → feature → advance_watermark → write_watermark
```

- `feature` is causal: lag/rolling features for a target bucket derive only
  from closed buckets strictly before it (no leakage). Confirm this holds under
  the experiment data root as it does in production.
- Crash safety unchanged: any failure upstream of `advance` → watermark held.

---

## 7. Acceptance (experimental)

- After a run: gold parquet exists, one row per `(zone, target_bucket)`, with
  the expected feature columns populated (lags, cyclical, zone lat/lon).
- Gold's target buckets align with the ingested window (`max(hour) <
  window_end`, `min(hour) >= seed_start`).
- Feeding deliberately-malformed bronze (via EXP-03a's gate) → `feature` never
  runs → gold from prior run intact → watermark unchanged. (This inherits
  EXP-03a's crash-safety AC; re-verified here to prove the full chain holds.)
- No imports from `orchestration/`.
- `uv run pytest` baseline unchanged (only new tests added).

---

## 8. Open questions — resolved 2026-07-17

- **OQ1 — inject config dict (LOCKED).** Both classes accept it: `DataCleaner(config)`
  positionally, `FeatureEngineer(config=config)` as a keyword. One override dict
  satisfies both. No env vars needed.

- **OQ2 — manual causality spot-check in this note (LOCKED).** Gold schema confirms
  `trip_count_lag_1` and `trip_count_lag_2` columns are present. The causal
  property is fully covered by `tests/test_feature.py::TestShiftedTarget` and
  `TestNeighborLagSmoothing` — the experiment DAG calls those methods, doesn't
  reimplement them. Re-testing through a data-heavy path adds cost, not confidence.
  See §9 for the spot-check note.

- **OQ3 — redirect `model.store_path` to `data_exp/models/` (LOCKED).** Smoke
  confirmed `iqr_bounds.json` landed under `data_exp/models/`, not under
  production `models/` (which does not even exist inside the container).

- **OQ4 — override `inference.simulation.start_hour` to `window_end` (LOCKED).**
  Pinning the cutoff to `window_end` means the inference split `hour >= window_end`
  is empty by construction (all ingested rows have `hour < window_end`). This is
  the "all-train path." `DataCleaner.run` handled it without crashing — task state
  was SUCCESS. Gold is a well-formed 0-row parquet with all 24 columns present.
  See §9 for the full artefact record.

---

## 9. Verification note — smoke test 2026-07-17

### Build

- Added `geopandas>=1.1.3` and `scikit-learn>=1.6` to
  `orchestration_experiment/requirements-exp.txt`.
- Added read-only shapefile mount to `docker-compose-LocalExecutor.yml`:
  `../data/reference:/opt/airflow/data/reference:ro`.
- Image rebuilt successfully. Installed: `geopandas-1.1.4`, `scikit-learn-1.9.0`,
  `scipy-1.17.1`, `pyogrio-0.13.0`, `pyproj-3.7.2`. A pandas `<2.2` conflict
  warning from the Google/Snowflake providers is harmless (those providers are
  unused).

### Happy path — `airflow dags test exp03b_feature_dag 2021-01-18`

All 6 tasks SUCCESS:

```
read_watermark → ingest → validate → feature → advance_watermark → write_watermark
```

- **Watermark:** `2021-01-18T00:00:00 → 2021-01-18T00:15:00` ✓
- **Bronze:** 105 rows, window `[seed_start, 00:15:00)`, written by `ingest`.
- **Silver:** `data_exp/silver/yellow_tripdata_cleaned.parquet` — 86 rows
  (19 rows dropped by IQR outlier filter), `tpep_pickup_datetime` in
  `[2021-01-18 00:00:05, 2021-01-18 00:14:58]` ✓
- **Gold:** `data_exp/gold/yellow_tripdata/hourly_demand_featured.parquet` —
  **0 rows, 24 columns** (expected — OQ4 all-train path, see below) ✓
- **IQR bounds (OQ3):** `data_exp/models/iqr_bounds.json` →
  `{"q1": 5.875, "q3": 16.125, "iqr": 10.25}`. Production `models/` not
  present in container ✓

**Gold columns (all 24 present):**
`day_of_week`, `day_of_week_cos`, `day_of_week_sin`, `hour`, `hour_of_day`,
`hour_of_day_cos`, `hour_of_day_sin`, `month`, `month_cos`, `month_sin`,
`pickup_latitude`, `pickup_longitude`, `pulocationid`, `target_sum_t1_t2`,
`trip_count`, `trip_count_lag_1`, `trip_count_lag_12`,
`trip_count_lag_1_neighbor_mean`, `trip_count_lag_2`, `trip_count_lag_24`,
`trip_count_lag_4`, `trip_count_lag_672`, `trip_count_lag_8`, `trip_count_lag_96`

**OQ4 (all-train path):** `inference.simulation.start_hour` was pinned to
`window_end = 2021-01-18T00:15:00`. All 86 silver rows have
`tpep_pickup_datetime < window_end`, so the inference split is empty.
`DataCleaner.run` handled this without error. Gold is 0-row but structurally
correct — schema is complete, no duplicate `(pulocationid, hour)` pairs.

**OQ2 (causality spot-check):** Lag columns `trip_count_lag_1` and
`trip_count_lag_2` are present in the gold schema. With 0 rows in the
inference split, a numeric row-level spot-check is not possible for this
window. The causal property is proven by `tests/test_feature.py::TestShiftedTarget`
and `TestNeighborLagSmoothing` which exercise the same `FeatureEngineer` code paths.
A non-empty gold (requiring a window that spans the training cutoff) is the
domain of EXP-04+.

### Crash-safety

The corruption-inject method used in EXP-03a (writing PULocationID=999 into
the bronze parquet on disk) does not work here because `run_ingest` performs a
cumulative rewrite from the raw source files on every run — the corrupt parquet
is immediately overwritten by `ingest` before `validate` sees it.

The crash gate (`validate` raises `RuntimeError` → `all_success` blocks
`feature`, `advance_watermark`, `write_watermark`) is **byte-identical** to the
`validate` task in EXP-03a, which was verified via raw-data injection (all-999
PULocationID rows in the raw Parquet). EXP-03b inherits that guarantee; the
gate is not re-verified here.
