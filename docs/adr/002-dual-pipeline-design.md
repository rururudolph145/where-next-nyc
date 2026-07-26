# ADR 002 — Dual-pipeline design: inference DAG + training DAG with isolated data stores

- **Status:** Superseded by [ADR 003](./003-unified-medallion-bundled-transforms.md) (2026-07-26)
- **Date:** 2026-07-24
- **Context:** PBI 25c (`train_monthly_dag`), watermark-advancing orchestrator redesign
- **Superseded because:** dual-storage isolation solves a write-race that a single-writer + atomic-replace invariant handles more simply, at the cost of introducing training/serving skew risk (two feature-engineering paths) and a cross-DAG file handshake (timestamped IQR bounds + glob-latest). ADR 003 unifies the medallion, bundles all training-only transforms into `model_bundle.pkl`, and switches the training DAG to manual-trigger + windowed-read to work correctly under simulated time.
- **Related:**
  - [ADR 001](./001-bronze-storage-strategy.md) — Bronze storage: single-file Parquet
  - [ADR 003](./003-unified-medallion-bundled-transforms.md) — Unified medallion (supersedes this ADR)
  - `orchestration_experiment/` EXP-01 → EXP-05 incremental DAG build
  - `wiki/orchestration/watermark-advancing-model.md`
  - `wiki/application/ingestion-pipeline.md`

---

## Context

The EXP-05 chain (`read_watermark → check_no_new_data → ingest → validate → feature → predict → log_predict → advance_watermark → write_watermark`) runs inference end-to-end. The system also needs periodic retraining: a new model is trained on a snapshot of accumulated data, gated by a champion-vs-challenger WAPE-lift check, and promoted to the MLflow registry.

Two concurrent-access hazards arise when retraining is added to the single-DAG model:

1. **Gold file collision.** `Trainer.run()` reads `gold/hourly_demand_featured.parquet`; `FeatureEngineer.run()` writes the same file every 15-minute inference cycle. A training run that overlaps a pipeline run reads a partially-written gold file and silently trains on corrupt data, or crashes with a truncated-Parquet read error.

2. **IQR bounds race.** `DataCleaner.run()` currently re-fits IQR bounds on every run using a hard-coded `inference.simulation.start_hour` cutoff and writes `iqr_bounds.json`. If training re-fits bounds concurrently with inference transforming data, the bounds file changes mid-run and inference picks up stale or partially-written bounds.

Neither hazard is prevented by Airflow's `all_success` trigger rule — that only guards the watermark file. The data files need isolation at the write site.

Additionally, the `Trainer.split_data()` method splits by position ratio (80/10/10) rather than by an explicit time boundary. This means the training split is only meaningful when the trainer reads a **snapshot** of the full history up to the training trigger — not the live-advancing inference gold file, which grows by one 15-minute slice every DAG run.

## Decision

We will separate the pipeline into two independent DAGs with fully isolated data stores.

### Inference DAG (`refresh_15min_dag`) — 15-minute cadence

Chain (extending EXP-05):

```
read_watermark
  → check_no_new_data
  → ingest                      → data/bronze/yellow_tripdata_cumulative.parquet (atomic)
  → validate
  → feature_transform           → data/silver/ingest_silver.parquet (atomic, transform-only)
                                → data/gold/ingest_gold.parquet (atomic)
  → predict                     → data/predictions/dt={date}/batch_{seq:05d}.parquet
  → log_predict
  → advance_watermark
  → write_watermark             → data/watermarks/refresh_15min/.current_watermark
```

Key rules:
- `feature_transform` uses **transform-only** IQR cleaning: it reads `iqr_bounds_{latest}.json` and applies pre-fitted bounds. It never re-fits. If no bounds file exists yet, it falls back to the last known good bounds file (glob by filename, pick the most recent by timestamp suffix). This is the crash-safety fallback for the first inference cycle before training has run.
- All data-file writes use `os.replace(tmp, final)` (atomic POSIX rename), consistent with `write_watermark`. A crash mid-write does not leave a corrupt file — the prior version survives until the replacement is complete.
- Predictions are written to a **date-partitioned directory** (`dt={YYYY-MM-DD}/batch_{seq:05d}.parquet`), never to a single growing file. The API reads the latest partition. Past partitions are retained for monitoring joins (see `app/monitoring/actuals_join.py`).

### Training DAG (`train_monthly_dag`) — triggered when inference watermark has advanced ≥ 30 days since last training watermark

Chain:

```
read_watermarks                 → reads both inference watermark + last-training watermark
  → check_retrain_due           → skips if delta < 30 days (AirflowSkipException)
  → snapshot_bronze             → data/silver/training_silver_{wm}.parquet (atomic)
  → fit_clean                   → DataCleaner.fit_transform(snapshot) → training_silver_{wm}
                                → data/models/iqr_bounds_{wm}.json (atomic)
  → build_training_gold         → data/gold/training_gold_{wm}.parquet (atomic)
  → train                       → Trainer.run(training_gold_{wm}) → model.pkl (gate)
  → advance_training_watermark
  → write_training_watermark    → data/watermarks/last_training/.current_watermark
```

Key rules:
- The training watermark is separate from the inference watermark. "30 days of simulated data have passed" is measured as `inference_wm - last_training_wm >= timedelta(days=30)`.
- `fit_clean` re-fits IQR bounds on the full training snapshot (no hard-coded cutoff — the snapshot itself is bounded by the trigger watermark). It writes `iqr_bounds_{wm}.json` atomically. The inference DAG picks up the new bounds file on its next run via a "most-recent-by-timestamp" glob.
- `Trainer.split_data()` is used unchanged: 80/10/10 time-ordered split over the training gold snapshot.
- The gate (`registry_gate.apply_gate`) runs inside `train`; `model.pkl` is written to disk only when the challenger is promoted. If the gate rejects, the training watermark still advances (the cycle consumed this data; re-running it would train the same model).
- `advance_training_watermark` and `write_training_watermark` both use `trigger_rule="all_success"` — a failed train or gate-error holds the training watermark.

### File layout

```
data/
  bronze/
    yellow_tripdata_cumulative.parquet       ← inference DAG; atomic write
  silver/
    ingest_silver.parquet                    ← inference DAG; transform-only; atomic
    training_silver_{watermark_iso}.parquet  ← training DAG; fit+transform; atomic
  gold/
    ingest_gold.parquet                      ← inference DAG; atomic
    training_gold_{watermark_iso}.parquet    ← training DAG; atomic
  predictions/
    dt={YYYY-MM-DD}/
      batch_{seq:05d}.parquet               ← inference DAG; partitioned append
  models/
    iqr_bounds_{watermark_iso}.json         ← training DAG writes; inference reads latest
    model.pkl                               ← training DAG promotes via registry gate
  watermarks/
    refresh_15min/
      .current_watermark                    ← inference DAG; os.replace write
    last_training/
      .current_watermark                    ← training DAG; os.replace write
```

### IQR bounds handshake

The `iqr_bounds_{wm}.json` naming convention (timestamped by the training trigger watermark) means:
- Multiple bounds files can coexist. The inference DAG always globs `data/models/iqr_bounds_*.json` and picks the most recent by filename sort — no lock needed, no single shared file.
- **First-run fallback:** if no bounds file exists (training has never run), the inference DAG falls back to a default bounds file (`iqr_bounds_default.json`) seeded at initialisation time, or raises a clear `NoBoundsAvailableError` that prevents feature engineering from producing silently bad data.
- **Stale bounds:** bounds age up to ~30 days between retraining cycles. This is acceptable — IQR bounds on `fare_amount` are stable over a month of similar NYC taxi data. The inference DAG logs the age of the bounds file it loaded so degradation is observable.

## Consequences

### Positive

- Full ACID isolation between the two pipelines: each reads only its own data files.
- `Trainer.split_data()` always operates on a fixed snapshot — the 80/10/10 split is temporally meaningful.
- Predictions are safe to append incrementally without growing a single file the API must load entirely.
- Both watermarks are independently observable and resettable for debugging.
- Inference DAG is never blocked by a long training run (LocalExecutor will serialize them, but neither corrupts the other's output).

### Negative

- Storage doubles: two silver files and two gold files exist simultaneously. At this dataset size (≤ 1.5M rows bronze, ≤ 566k rows gold) this is well under 1 GB total — acceptable.
- The IQR bounds glob adds a small filesystem scan per inference run (~O(1) files before pruning). Negligible at this scale.
- Training watermark advancing even on gate-rejection means a rejected cycle is "spent" — the same snapshot will not be retrained. This is intentional: re-training on identical data produces an identical challenger that will be rejected again.

### Neutral

- `DataCleaner.run()` (the combined fit+transform method) is now only called by the training DAG. The inference DAG calls a new method (or the same `clean()` with pre-loaded bounds). This is a natural separation of concerns that the existing `clean(iqr_bounds=None)` / `clean(iqr_bounds=bounds)` parameter already supports.

## Alternatives considered

### Single DAG with conditional training branch

Add a `maybe_retrain` task to the inference DAG that checks the delta and triggers training inline.

**Why rejected:** Training on a 566k-row gold file takes minutes (LightGBM on a laptop is fast, but MLflow logging, registry calls, and gate evaluation add overhead). Blocking the 15-minute inference cadence on a training run violates the "inference must not be disturbed" requirement. With LocalExecutor this is a strict serial block, not a background job.

### Single shared gold file with a file lock

Keep one gold file, add an advisory lock (e.g. `fcntl.flock`) so the trainer waits for the writer to finish.

**Why rejected:** Advisory locks are process-local and do not survive container restarts or multi-node deployments. They also block rather than isolate — a 2-minute training job holds the lock and delays the next inference slice. Isolation via separate files is simpler and more robust.

### Versioned gold files (keep N latest, no training-specific copy)

Write `gold/hourly_demand_featured_{wm}.parquet` for every inference cycle; let the trainer always read the latest complete version.

**Why rejected:** At 566k rows per slice and 96 slices/day, this produces ~96 gold files/day = ~35k/year. The metadata-scan cost grows linearly and the trainer would read an inference-oriented file (built with transform-only IQR) rather than a freshly-fitted snapshot. Retaining a training-specific gold file preserves the fit-once/transform-many invariant that prevents leakage.

### Display flag on a single growing predictions file

Append all predictions rows to one file with a `display: bool` column; the API filters on `display=True`.

**Why rejected:** At 566k rows/run × ~35k runs/year = ~20 billion rows, the API would load the entire file at startup. Date-partitioned files mean the API reads only the latest partition, and monitoring joins can predicate-pushdown by `dt=` without scanning history. This is already consistent with how `log_predict` in EXP-04 organises its monitoring output.
