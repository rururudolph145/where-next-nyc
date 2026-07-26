# ADR 003 — Unified medallion with bundled model transforms and manually-triggered windowed retraining

- **Status:** Accepted
- **Date:** 2026-07-26
- **Supersedes:** [ADR 002](./002-dual-pipeline-design.md) — Dual-pipeline design with isolated data stores
- **Context:** PBI 25c (`train_monthly_dag`) revised; watermark-advancing orchestrator design; simulated-time constraint
- **Related:**
  - [ADR 001](./001-bronze-storage-strategy.md) — Bronze storage: single-file Parquet
  - `orchestration_experiment/` EXP-01 → EXP-05 incremental DAG build

---

## Context

ADR 002 was accepted on 2026-07-24 to prevent concurrent-write hazards between a 15-minute inference DAG and a periodic training DAG. Its solution was **storage isolation** — dual silver files, dual gold files, timestamped IQR-bounds files with a glob-latest handshake between the two DAGs.

On review, that solution solves the concurrency problem by splitting the storage layer, but it introduces three costs that are load-bearing on future maintenance:

1. **Training/serving skew risk.** With two independent feature engineering paths writing two independent gold files (`ingest_gold.parquet` vs `training_gold_{wm}.parquet`), any drift between the two implementations — a bug fix applied to one and not the other, a subtle pandas or numpy version difference, different NaN handling — produces a model trained on features that do not match what it is served at inference time. This is the classic MLOps failure mode ("training-serving skew"), and it is the exact problem the industry moved to unified feature stores to solve.
2. **A cross-DAG file handshake with real cognitive cost.** ADR 002 introduced `iqr_bounds_{wm}.json` timestamped families, glob-latest resolution, first-run fallback (`iqr_bounds_default.json`), and an explicit "which bounds file did I load?" observability concern. Each of these is a coordination protocol between the two DAGs — extra moving parts, a subtle bug source, and additional concepts to explain in the README.
3. **Reaching for the wrong lever.** The original write-race hazard was a **Python-layer contract problem** (`DataCleaner.run()` re-fitting IQR bounds on every inference call; `FeatureEngineer.run()` and `Trainer.run()` targeting the same file with no coordination). ADR 002 solved it by splitting the *storage layer*. The simpler fix is to fix the Python contract: enforce a single-writer invariant on gold, and make inference-path `DataCleaner` transform-only.

Additionally, ADR 002 specified a **scheduled** retraining trigger (`Δwatermark ≥ 30 days` gates whether the training DAG runs). Under this project's simulated-time model (`DEMO_MODE` compresses months into minutes), scheduled retraining is meaningless — training itself takes real minutes, but the simulated clock advances so fast that "cadence" becomes undefined. Retraining must be **manually triggered** with a **deterministic data window**, so that training runs on human-controlled wall-clock intent while training *sees* a reproducible simulated-time slice.

## Decision

We will use **one unified medallion pipeline** (bronze → silver → gold), with a **single writer** (the inference DAG), **all training-only operations bundled inside the model artifact**, and **manually-triggered windowed retraining**.

### Core invariants

1. **Single writer to gold.** Only `refresh_15min_dag` writes to `bronze.parquet`, `silver.parquet`, and `gold.parquet`. The training DAG **never writes to those files** — it is a pure reader.
2. **Atomic replace on every data write.** All writes to bronze / silver / gold / predictions / model bundle / watermark files use `os.replace(tmp, final)` (POSIX atomic rename). A reader that opens a file mid-write either sees the fully-written prior version or the fully-written new version — never a torn file.
3. **Transform-only inference.** `DataCleaner.clean(data, iqr_bounds=loaded_bounds)` is called with pre-fitted bounds loaded from the model bundle. Inference **never fits** anything. This eliminates the EXP-05 "IQR re-fit on every run" bug at the contract level, not the storage level.
4. **Fit-in-training, bundled with the model.** All fit operations (target outlier thresholds, feature scaler μ/σ, IQR bounds) happen inside the training DAG's Python process, on the in-memory train split only. They are saved as **a single artifact** (`model_bundle.pkl`) containing `{"model": ..., "scaler": ..., "iqr_bounds": ..., "feature_names": [...], "trained_at_watermark": ...}`. Impossible for the model and its transforms to desync.
5. **Two watermarks, both single-writer.** `refresh_15min/.current_watermark` (written by inference) and `last_training/.current_watermark` (written by training) remain separate. These are DAG-progress markers, not storage duplication.
6. **Manual training trigger, windowed data read.** The training DAG has `schedule=None`. When manually triggered, it reads gold filtered to `[start, last_training_watermark + timedelta(days=30))` — not the full inference horizon. This makes retraining reproducible and independent of when in wall-clock the user clicks the trigger.

### File layout

```
data/
  bronze/
    yellow_tripdata_cumulative.parquet   ← inference DAG; atomic write; ONLY writer
  silver/
    silver.parquet                        ← inference DAG; transform-only cleaning; atomic
  gold/
    gold.parquet                          ← inference DAG; unified feature table; atomic
  predictions/
    dt={YYYY-MM-DD}/
      batch_{seq:05d}.parquet             ← inference DAG; date-partitioned append
  models/
    model_bundle.pkl                      ← training DAG; atomic replace on promotion
                                             contains: {model, scaler, iqr_bounds,
                                                        feature_names, trained_at_watermark}
  watermarks/
    refresh_15min/.current_watermark      ← inference DAG
    last_training/.current_watermark      ← training DAG
```

**Removed from ADR 002:** `training_silver_{wm}.parquet`, `training_gold_{wm}.parquet`, `iqr_bounds_{wm}.json` timestamped family, `iqr_bounds_default.json` first-run fallback, and the glob-latest bounds resolution logic.

### Inference DAG (`refresh_15min_dag`) — 15-minute cadence

```
read_watermark
  → check_no_new_data                    (skip if dataset exhausted)
  → ingest                               → bronze.parquet     (cumulative rewrite; atomic)
  → validate                              (Pandera bronze schema)
  → feature_transform                    (loads model_bundle.pkl → iqr_bounds)
                                         → silver.parquet    (clean() in transform-only mode; atomic)
                                         → gold.parquet      (FeatureEngineer.run(); atomic)
  → predict                              (loads model_bundle → scale + predict)
                                         → predictions/dt=.../batch_NN.parquet
  → log_predict
  → advance_watermark
  → write_watermark
```

### Training DAG (`train_monthly_dag`) — manual trigger only

```
read_watermarks                          (inference_wm + last_training_wm)
  → check_window_ready                   (skip if inference_wm < last_training_wm + 30d)
  → compute_window                       (window_end = last_training_wm + 30d)
  → train_in_memory:                     (single Python task; no intermediate files)
      1. Read gold.parquet ONCE, FILTER to [start, window_end)
      2. Time-ordered split (80/10/10) on the filtered slice
      3. Outlier removal on train split's target only
      4. Fit IQR bounds on train split features
      5. Apply IQR + fit scaler
      6. Train LightGBM
      7. Champion-vs-challenger WAPE-lift gate
      8. If promoted: bundle {model, scaler, iqr_bounds, feature_names, trained_at_watermark}
                     → os.replace(model_bundle.pkl.tmp, model_bundle.pkl)
  → advance_training_watermark           (to window_end; even on gate reject)
  → write_training_watermark
```

**Trigger semantics:**
- `schedule=None`. User triggers manually via Airflow UI or CLI.
- `check_window_ready` raises `AirflowSkipException` if `inference_wm < last_training_wm + 30d`. The DAG succeeds-with-skip when triggered too early — no wasted human attention, no partial writes.
- `advance_training_watermark` advances to `last_training_wm + 30d`, **not** to `refresh_watermark`. This preserves "each training cycle covers exactly one 30-day window" as a hard invariant, and enables manual walking of the simulated timeline slice by slice.

### `DataCleaner` API contract

```python
# Training path (fit + transform, in-memory only, bounds returned to caller):
bounds = DataCleaner.fit_bounds(train_split)
train_cleaned = DataCleaner.clean(train_split, iqr_bounds=bounds)

# Inference path (transform-only, bounds loaded from the model bundle):
bundle = joblib.load("model_bundle.pkl")
silver = DataCleaner.clean(bronze, iqr_bounds=bundle["iqr_bounds"])
```

The existing `clean(iqr_bounds=None)` / `clean(iqr_bounds=bounds)` parameter already supports this — the change is a contract clarification and a caller-site refactor, not a rewrite of `DataCleaner` itself.

## Consequences

### Positive

- **No training/serving skew by construction.** One feature-engineering path, one gold file, one set of transforms bundled inside the model. Impossible for training features and inference features to diverge — they come from the same function on the same file.
- **Storage stays as flat as ADR 001 designed it.** Bronze / silver / gold each remain a single file, atomic-replace on write. No timestamped families, no glob-latest.
- **One cross-DAG artifact instead of several.** `model_bundle.pkl` carries the model, scaler, IQR bounds, feature names, and training watermark as one atomic object. Impossible for them to desync.
- **Manual training trigger with windowed data reads** eliminates the "how do we schedule retraining under simulated time?" problem entirely. Training runs on wall-clock human intent; training sees a deterministic simulated-time window.
- **Reproducibility for free.** Training twice at the same `last_training_watermark` reads the exact same slice of gold and produces the exact same challenger, regardless of when the user clicks "trigger."
- **Bootstrap is loud, not silent.** First-run inference fails until a model bundle exists. No default-bounds fallback that could mask real problems.
- **Replayable demo.** Reset both watermarks to zero, click the training trigger repeatedly, and the training DAG walks the simulated timeline forward one 30-day slice at a time, producing a sequence of model bundles that can be diffed.

### Negative / Accepted tradeoffs

- **Bootstrap requires manual sequencing.** The training DAG must be triggered once before inference can serve. Documented in the README; a one-time cost per environment.
- **`DataCleaner` caller-site refactor.** Existing callers of `DataCleaner.run()` must be audited: training-path callers keep fit + transform, inference-path callers switch to `clean(data, iqr_bounds=loaded)`. Mechanical but not free.
- **Training reads gold live, not a snapshot.** `os.replace` guarantees no torn reads, but "exactly which bytes did we train on?" is not captured as a persisted artifact. If reproducibility at that level becomes a hard requirement, an optional `snapshot_gold` first task can be added later. Deferred as YAGNI.
- **Model bundle grows slightly.** `model_bundle.pkl` now contains the model plus a small dict of transforms and metadata. Still well under 10 MB. Negligible.

### Neutral

- **Two watermark files remain.** Unchanged from ADR 002 — these are DAG-progress markers, not storage duplication.
- **Predictions layout unchanged.** `predictions/dt={date}/batch_{seq:05d}.parquet` stays as ADR 002 defined it. That was never storage duplication — it is the correct incremental-append shape for time-series output.

## Edge cases and their handling

**First run (no `model_bundle.pkl` yet).** Inference fails loudly (`ModelBundleNotFoundError`) at `feature_transform` and `predict`. Watermark does not advance. User manually triggers training once → produces first bundle → inference catches up.

**Training reads gold while inference is mid-write.** `os.replace` is a POSIX atomic rename. The training DAG either sees the old fully-valid file or the new fully-valid file; no torn reads. Under manual triggering, the user simply avoids clicking during a 15-minute slice.

**Training gate rejects the challenger.** `model_bundle.pkl` is not replaced. Training watermark still advances to `window_end`. Rationale: retraining on identical data produces an identical rejection; advancing prevents an infinite retry loop on stagnant data.

**Training crashes mid-fit.** All fitting is in-memory; no partial file writes. Training watermark not advanced. Next manual trigger re-attempts the same window. Idempotent.

**Training crashes during `os.replace` of the bundle.** Kernel-atomic; either the swap completed or it didn't. No corrupt-bundle state possible. A leftover `.tmp` file can be manually deleted or ignored (next run overwrites).

**Schema drift — gold has new columns since last training.** Bundle stores `feature_names` explicitly. Inference selects columns by name: missing column → `KeyError` (fail loud); extra column → silently ignored (correct: model can't use features it wasn't trained on).

**Manual training triggered too early.** `check_window_ready` raises `AirflowSkipException` with a clear message (`"inference_wm=X, need >= last_training_wm + 30d = Y"`). DAG shows as skipped in Airflow UI. No files touched, no watermark advanced.

**Training watermark ahead of inference watermark.** Impossible under normal operation. `check_window_ready` asserts `refresh_watermark >= last_training_watermark` and raises `WatermarkInvariantError` if violated. Fail loud, do not recover.

## Alternatives considered

**Dual data-store isolation (ADR 002 as originally accepted).** Solves a write-race that does not exist under `max_active_runs=1` + atomic replace, at the cost of doubling storage, introducing an IQR-bounds handshake, and creating a persistent training/serving skew risk. Rejected: fixes the symptom (concurrent writes) by splitting storage rather than fixing the cause (bad `DataCleaner.run()` contract and dual writers to gold).

**Single DAG with conditional retrain branch.** Rejected for the same reasons ADR 002 rejected it: training runtime blocks the 15-minute inference cadence under LocalExecutor, and manual triggering is impossible when the two are fused.

**Scheduled retraining triggered by watermark delta (as in ADR 002).** Rejected because simulated time makes "cadence" meaningless: training takes real minutes, but the simulated clock advances at `DEMO_MODE` speedup. Manual trigger + windowed read decouples "when training runs" (wall-clock) from "what training sees" (simulated-time window).

**Table format (Delta / Iceberg via PyIceberg).** Deferred as in ADR 001. The demo does not need transactional snapshots; single-writer + atomic replace is sufficient. Table formats become the right answer only when serving and compaction run concurrently on the same files — not this workload.

**In-memory snapshot copy (`training_gold_snapshot_{wm}.parquet`).** Deferred. Reproducibility of "exactly which bytes did we train on?" is nice-to-have but not required for a portfolio demo. If needed later, it is a single new task in the training DAG — no ADR revision required.

## Migration from ADR 002

The following ADR-002 elements are removed by this ADR:

| ADR 002 element | Status under ADR 003 |
|---|---|
| Dual silver files (`ingest_silver` + `training_silver_{wm}`) | Removed — unified silver, single writer |
| Dual gold files (`ingest_gold` + `training_gold_{wm}`) | Removed — unified gold, single writer |
| `iqr_bounds_{wm}.json` timestamped family + glob-latest | Removed — bounds live inside `model_bundle.pkl` |
| `iqr_bounds_default.json` first-run fallback | Removed — first-run fails loudly |
| Watermark-delta scheduled retrain trigger | Removed — manual trigger + windowed read |
| Dual watermarks (inference + training) | Kept — DAG-progress markers |
| Date-partitioned predictions | Kept — correct time-series append pattern |
| Atomic `os.replace` for all writes | Kept and reinforced — now the only concurrency-safety mechanism needed |
| Training gate advances watermark even on reject | Kept — prevents infinite retrain loop on stagnant data |
| `Trainer.split_data()` on a bounded snapshot | Kept semantically — snapshot is now a filtered in-memory read, not a separate file |
